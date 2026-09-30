import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_


class GRU4Rec(nn.Module):
    """
    Session-based Recommendation with GRU (GRU4Rec).

    Hidasi and Karatzoglou, RecSys '18.
    参考实现: GAMER (SeqRec/models/discriminative/GRU4Rec/model.py,
    https://github.com/wzf2000/GAMER), 其实现参考 RecBole gru4rec。

    结构与 GAMER/RecBole 一致:
    物品嵌入 -> dropout -> GRU(bias=False) -> dense -> 取最后一个
    真实位置(按 seq_len - 1 gather)的隐状态作为序列表示 z。

    打分遵循 GAMER SeqModel 判别式基线的共享物品嵌入点积
    (z @ E^T), 训练目标对齐 SIDReasoner(KDD'26)附录 A 的基线
    协议: 单目标 + 均匀采样负例的 Binary Cross-Entropy
    (bce_loss, 默认); 全词表 CrossEntropy(ce_loss)保留供对照,
    但其结果会显著高于论文报告值(见 runs/ 与 docs)。

    n_items 为物品总数, 物品 id 从 0 开始,
    embedding 第 0 行保留给 padding(右侧补 0, 与 GAMER 的
    TraditionalCollator padding_side='right' 一致)。
    """

    def __init__(
        self,
        n_items,
        embedding_size=64,
        hidden_size=128,
        n_layers=1,
        dropout=0.3,
    ):
        super().__init__()
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.item_embeddings = nn.Embedding(
            n_items + 1, embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.gru_layers = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def gather_indexes(self, output, gather_index):
        """按 gather_index 收取每个样本指定时间步的隐状态。"""
        gather_index = gather_index.view(-1, 1, 1).expand(
            -1, -1, output.shape[-1]
        )
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        """
        item_seq: [B, L] 右侧补 0 的序列
        item_seq_len: [B] 每个序列的真实长度
        返回序列表示 z: [B, embedding_size]
        """
        item_seq_emb = self.item_embeddings(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def full_scores(self, item_seq, item_seq_len):
        """
        一次前向计算所有物品得分(用于全量排序评估),
        共享物品嵌入点积打分。返回 [B, n_items], 第 i 列对应物品 id i。
        """
        z = self.forward(item_seq, item_seq_len)
        emb = self.item_embeddings.weight[1:]
        return z @ emb.t()

    def ce_loss(self, item_seq, item_seq_len, target):
        """
        全词表 CrossEntropy(与 GAMER SeqModel 的 CE 损失一致)。
        target: [B], 物品 id(0 起始)。
        """
        z = self.forward(item_seq, item_seq_len)
        emb = self.item_embeddings.weight[1:]
        logits = z @ emb.t()
        return F.cross_entropy(logits, target)

    def _sample_negatives(self, target):
        """均匀采样与 target 不冲突的负例(MiniOneRec 惯例: 仅避开目标)。"""
        neg = torch.randint(0, self.n_items, target.shape, device=target.device)
        clash = neg == target
        while clash.any():
            neg = torch.where(
                clash,
                torch.randint(0, self.n_items, neg.shape, device=target.device),
                neg,
            )
            clash = neg == target
        return neg

    def bce_loss(self, item_seq, item_seq_len, target, n_neg=1):
        """
        单目标 Binary Cross-Entropy + n_neg 个均匀采样负例
        (SIDReasoner 附录 A 基线协议)。
        target: [B], 物品 id(0 起始)。
        """
        z = self.forward(item_seq, item_seq_len)
        emb = self.item_embeddings.weight[1:]
        losses = []
        for _ in range(n_neg):
            neg = self._sample_negatives(target)
            pos = (z * emb[target]).sum(-1)
            ng = (z * emb[neg]).sum(-1)
            losses.append(
                F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            )
            losses.append(
                F.binary_cross_entropy_with_logits(ng, torch.zeros_like(ng))
            )
        return torch.stack(losses).mean()
