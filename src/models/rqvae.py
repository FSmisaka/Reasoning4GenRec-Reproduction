"""
残差量化自编码器(RQ-VAE)

TIGER(Recommender Systems with Generative Retrieval)一脉的语义 ID
生成器: MLP 编码器 -> num_levels 层残差向量量化(每层 codebook_size 个
code, EMA 更新 + 可选 k-means 初始化) -> MLP 解码器, 损失为
重建 MSE + commitment。

输出侧与消费端约定对齐: 每个物品量化为 num_levels 个 code, 即
<a_N><b_N><c_N>(num_levels=3, codebook_size=256)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_sq_dist(x, y):
    """(N, D) x (M, D) -> (N, M) 平方欧氏距离(开根号与否不影响 argmin)。"""
    return (
        (x * x).sum(dim=1, keepdim=True)
        - 2 * x @ y.t()
        + (y * y).sum(dim=1)
    ).clamp_min_(0)


def kmeans(x, n_clusters, iters=50):
    """朴素 Lloyd k-means(物品数仅数千, 全量内存计算即可)。

    x: (N, D); 返回 (n_clusters, D) 中心。空簇用随机样本重播种。
    依赖全局随机态, 调用前应已 set_seed。
    """
    n = x.shape[0]
    perm = torch.randperm(n, device=x.device)[:n_clusters]
    centers = x[perm].clone()
    for _ in range(iters):
        assign = pairwise_sq_dist(x, centers).argmin(dim=1)
        counts = torch.bincount(assign, minlength=n_clusters)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assign, x)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        dead = (~nonempty).nonzero(as_tuple=True)[0]
        if dead.numel():
            centers[dead] = x[
                torch.randperm(n, device=x.device)[: dead.numel()]
            ]
    return centers


class ResidualVQLayer(nn.Module):
    """单层向量量化: 最近邻查表 + EMA 码本更新 + straight-through。"""

    def __init__(self, codebook_size, dim, ema_decay=0.99,
                 commitment_beta=0.25, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.commitment_beta = commitment_beta
        self.eps = eps
        self.embed = nn.Parameter(
            torch.randn(codebook_size, dim) * 0.02, requires_grad=False
        )
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", self.embed.data.clone())
        self.register_buffer("inited", torch.tensor(False))

    @torch.no_grad()
    def init_from_data(self, data, kmeans_iters=50):
        """用一组(残差)向量做 k-means 初始化码本。"""
        centers = kmeans(data, self.codebook_size, kmeans_iters)
        self.embed.copy_(centers)
        self.embed_avg.copy_(centers)
        self.cluster_size.fill_(1.0)
        self.inited.fill_(True)

    def assign(self, x):
        """返回最近邻 code 序号, (B,)。"""
        return pairwise_sq_dist(x, self.embed).argmin(dim=1)

    @torch.no_grad()
    def _ema_update(self, x, codes):
        onehot = F.one_hot(codes, self.codebook_size).to(x.dtype)
        cluster_sum = onehot.sum(dim=0)
        embed_sum = onehot.t() @ x
        decay = self.ema_decay
        self.cluster_size.mul_(decay).add_(cluster_sum, alpha=1 - decay)
        self.embed_avg.mul_(decay).add_(embed_sum, alpha=1 - decay)
        # 重播种近乎死亡的 code(EMA 使用量 < 0.3):
        # 用 batch 内随机样本替换, 防止码本坍缩。
        dead = self.cluster_size < 0.3
        if bool(dead.any()):
            pick = torch.randint(
                0, x.shape[0], (int(dead.sum()),), device=x.device
            )
            self.embed[dead] = x[pick]
            self.embed_avg[dead] = x[pick]
            self.cluster_size[dead] = 1.0
        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.eps)
            / (n + self.codebook_size * self.eps)
            * n
        )
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

    def forward(self, x):
        """x: (B, D) 当前残差。

        返回 (quantized, codes, commit_loss):
        - quantized 已做 straight-through, 梯度直通编码器;
        - commit_loss 为 commitment 损失(系数 beta 已乘)。
        """
        codes = self.assign(x)
        quantized = F.embedding(codes, self.embed)
        if self.training:
            self._ema_update(x, codes)
        commit_loss = F.mse_loss(quantized.detach(), x) * self.commitment_beta
        quantized = x + (quantized - x).detach()
        return quantized, codes, commit_loss


class RQVAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=(512,), code_dim=256,
                 codebook_size=256, num_levels=3, ema_decay=0.99,
                 commitment_beta=0.25, kmeans_init=True, kmeans_iters=50):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        dims = [input_dim, *hidden_dims, code_dim]
        encoder, decoder = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            encoder.append(nn.Linear(a, b))
            encoder.append(nn.GELU())
            decoder.append(nn.Linear(b, a))
            decoder.append(nn.GELU())
        encoder.pop()  # 量化前的隐层不需要激活
        decoder.pop()
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*reversed(decoder))
        self.levels = nn.ModuleList(
            ResidualVQLayer(codebook_size, code_dim, ema_decay,
                            commitment_beta)
            for _ in range(num_levels)
        )

    @torch.no_grad()
    def init_codebooks(self, x):
        """逐层贪心 k-means 初始化: 第 l 层在前 l-1 层量化后的残差上聚类。"""
        residual = self.encoder(x)
        for level in self.levels:
            if not bool(level.inited):
                level.init_from_data(residual, self.kmeans_iters)
            residual = residual - F.embedding(level.assign(residual),
                                              level.embed)

    def forward(self, x):
        z_e = self.encoder(x)
        residual = z_e
        quantized_sum = 0
        commit_loss = z_e.new_zeros(())
        codes = []
        for level in self.levels:
            quantized, code, commit = level(residual)
            codes.append(code)
            commit_loss = commit_loss + commit
            residual = residual - quantized.detach()
            quantized_sum = quantized_sum + quantized
        x_recon = self.decoder(quantized_sum)
        recon_loss = F.mse_loss(x_recon, x)
        return recon_loss + commit_loss, {
            "recon": recon_loss.detach(),
            "commit": commit_loss.detach(),
            "codes": torch.stack(codes, dim=1),
        }

    @torch.no_grad()
    def quantize(self, x):
        """推理期量化: 返回 (B, num_levels) 的 code 序列。"""
        residual = self.encoder(x)
        codes = []
        for level in self.levels:
            code = level.assign(residual)
            codes.append(code)
            residual = residual - F.embedding(code, level.embed)
        return torch.stack(codes, dim=1)
