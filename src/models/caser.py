import torch
from torch import nn
from torch.nn import functional as F

activation_getter = {
    "iden": lambda x: x,
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigm": torch.sigmoid,
}


class Caser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser).

    Personalized Top-N Sequential Recommendation via Convolutional
    Sequence Embedding, Jiaxi Tang and Ke Wang, WSDM '18.
    原始实现: https://github.com/graytowne/caser_pytorch (caser.py)

    num_items 应为 物品数 + 1: 物品 id 从 1 开始, 0 保留给序列 padding
    (与原始实现 to_sequence 中的 +1 偏移一致)。
    num_users 应为 用户数 + 1: 用户 id 从 1 开始, 0 保留给
    验证/测试中未在训练集出现过的用户。
    """

    def __init__(
        self,
        num_users,
        num_items,
        L=5,
        d=50,
        nv=4,
        nh=16,
        drop=0.5,
        ac_conv="relu",
        ac_fc="relu",
    ):
        super().__init__()
        self.L = L
        self.d = d
        self.n_h = nh
        self.n_v = nv
        self.drop_ratio = drop
        self.ac_conv = activation_getter[ac_conv]
        self.ac_fc = activation_getter[ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, d)
        self.item_embeddings = nn.Embedding(num_items, d)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(1, self.n_h, (i, d)) for i in lengths]
        )

        # fully-connected layer
        self.fc1_dim_v = self.n_v * d
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, d)
        # W2, b2 are encoded with nn.Embedding, as we don't need to
        # compute scores for all items
        self.W2 = nn.Embedding(num_items, d + d)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(
            0, 1.0 / self.user_embeddings.embedding_dim
        )
        self.item_embeddings.weight.data.normal_(
            0, 1.0 / self.item_embeddings.embedding_dim
        )
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        """
        给定 (sequence, user, targets) 三元组计算推荐得分,
        与原始实现的 forward 保持一致。

        seq_var:   [B, L] 序列(0 为 padding)
        user_var:  [B, 1] 用户
        item_var:  [B, n] 待打分物品(训练时为 targets + negatives)
        for_pred:  评估时单用户对所有物品打分
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)
        user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res

    def full_scores(self, seq_var, user_var):
        """
        一次前向计算所有物品得分(用于全量排序评估), 数学上等价于
        原始 predict() 中对每个用户传入全部 item_ids 的做法。
        返回 [B, num_items - 1], 第 i 列对应 0 起始的原始物品 id i
        (embedding 第 0 槽位是 padding, 已去掉)。
        """
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)
        user_emb = self.user_embeddings(user_var).squeeze(1)

        out_v = self.conv_v(item_embs)
        out_v = out_v.view(-1, self.fc1_dim_v)

        out_hs = list()
        for conv in self.conv_h:
            conv_out = self.ac_conv(conv(item_embs).squeeze(3))
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)

        out = torch.cat([out_v, out_h], 1)
        out = self.dropout(out)
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)

        scores = (
            torch.matmul(x, self.W2.weight.t())
            + self.b2.weight.squeeze(-1)
        )
        return scores[:, 1:]
