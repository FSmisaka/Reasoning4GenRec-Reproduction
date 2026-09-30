import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, eps=1e-12):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d_model, d_model)
        self.LayerNorm = nn.LayerNorm(d_model, eps=eps)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * math.sqrt(
            1.0 / self.head_dim
        )
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.dense(ctx)
        out = self.out_dropout(out)
        return self.LayerNorm(out + x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout, eps=1e-12):
        super().__init__()
        self.dense_1 = nn.Linear(d_model, d_inner)
        self.dense_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        out = self.dense_2(torch.relu(self.dense_1(x)))
        out = self.dropout(out)
        return self.LayerNorm(out + x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, dropout, eps=1e-12):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, eps)
        self.ffn = FeedForward(d_model, d_inner, dropout, eps)

    def forward(self, x, attn_mask):
        return self.ffn(self.attn(x, attn_mask))


class SASRec(nn.Module):
    def __init__(
        self,
        n_items,
        max_seq_len=10,
        hidden=128,
        n_layers=2,
        n_heads=2,
        d_inner=256,
        dropout=0.5,
        init_std=0.02,
        layer_norm_eps=1e-12,
        neg_resample_rounds=16,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.init_std = init_std
        self.neg_resample_rounds = neg_resample_rounds
        self.item_embedding = nn.Embedding(
            n_items + 1, hidden, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden, n_heads, d_inner, dropout, layer_norm_eps)
                for _ in range(n_layers)
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _attn_mask(self, seq):
        pad_mask = seq != 0
        mask = pad_mask[:, None, None, :].expand(-1, -1, seq.size(1), -1)
        mask = torch.tril(mask)
        dtype = torch.float32
        mask = (1.0 - mask.to(dtype)) * torch.finfo(dtype).min
        return mask

    def encode(self, seq):
        pos = torch.arange(seq.size(1), device=seq.device)
        x = self.item_embedding(seq) + self.position_embedding(pos)[None]
        x = self.LayerNorm(x)
        x = self.dropout(x)
        attn_mask = self._attn_mask(seq)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

    def last_hidden(self, seq):
        x = self.encode(seq)
        return x[:, -1]

    def full_scores(self, seq):
        h = self.last_hidden(seq)
        emb = self.item_embedding.weight[1:]
        return h @ emb.t()

    def bce_loss(self, seq, labels, exclude_mask):
        x = self.encode(seq)
        emb = self.item_embedding.weight
        pos_scores = torch.einsum("blh,blh->bl", x, emb[labels])
        neg = self._sample_negatives(exclude_mask, labels.shape)
        neg_scores = torch.einsum("blh,blh->bl", x, emb[neg])
        valid = (seq != 0).float()
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores), reduction="none"
        )
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores), reduction="none"
        )
        loss = (pos_loss + neg_loss) * valid
        return loss.sum() / valid.sum()

    def _sample_negatives(self, exclude_mask, shape):
        n = exclude_mask.size(1)
        neg = torch.randint(1, n, shape, device=exclude_mask.device)
        for _ in range(self.neg_resample_rounds):
            bad = exclude_mask.gather(1, neg)
            if not bad.any():
                break
            neg = torch.where(
                bad,
                torch.randint(1, n, neg.shape, device=exclude_mask.device),
                neg,
            )
        return neg

    def ce_loss(self, seq, labels):
        x = self.encode(seq)
        emb = self.item_embedding.weight[1:]
        logits = torch.einsum("blh,nh->bln", x, emb)
        valid = (seq != 0).float()
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.n_items),
            (labels - 1).clamp(min=0).reshape(-1),
            reduction="none",
        ).view(seq.size(0), -1)
        return (losses * valid).sum() / valid.sum()

    def bce_last_loss(self, seq, labels, n_neg=1):
        """
        单目标 Binary Cross-Entropy + n_neg 个均匀采样负例, 仅在
        最后位置监督(SIDReasoner 附录 A 基线协议 / MiniOneRec 风格;
        区别于原版 SASRec 的全位置 bce_loss)。
        seq/labels 均为 1 起始 id(0 为 padding)。
        """
        x = self.encode(seq)
        h = x[:, -1]
        emb = self.item_embedding.weight
        tgt = labels[:, -1]
        losses = []
        for _ in range(n_neg):
            neg = torch.randint(1, self.n_items + 1, tgt.shape, device=tgt.device)
            clash = neg == tgt
            while clash.any():
                neg = torch.where(
                    clash,
                    torch.randint(
                        1, self.n_items + 1, neg.shape, device=neg.device
                    ),
                    neg,
                )
                clash = neg == tgt
            pos = (h * emb[tgt]).sum(-1)
            ng = (h * emb[neg]).sum(-1)
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    pos, torch.ones_like(pos)
                )
            )
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    ng, torch.zeros_like(ng)
                )
            )
        return torch.stack(losses).mean()
