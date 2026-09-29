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

    卷积序列编码器(垂直/水平卷积 + 全连接层)与原始实现一致;
    打分与损失遵循 GAMER(SeqRec.modules.model_base.seq_model.SeqModel)
    对判别式序列基线的统一协议:
    - 无用户嵌入: 该数据生态中过半评估用户未在训练集出现,
      用户个性化无法泛化(GAMER 的判别式基线均为用户无关模型);
    - 共享物品嵌入打分 z @ E^T, 不使用原始的逐物品自由参数 W2/b2:
      本数据集每物品正样本监督仅 ~13 次, W2/b2 + 均匀负采样 BCE
      会使热门(常为目标)物品被系统性压低, 排序坍缩为反流行度;
    - 全词表 CrossEntropy 训练(等价于无穷负采样的对称更新)。

    num_items 为物品总数, 物品 id 从 0 开始;
    embedding 第 0 行保留给序列 padding(左侧补 0)。
    """

    def __init__(
        self,
        n_items,
        L=10,
        d=50,
        nv=4,
        nh=16,
        drop=0.5,
        ac_conv="relu",
        ac_fc="relu",
    ):
        super().__init__()
        self.n_items = n_items
        self.L = L
        self.d = d
        self.n_h = nh
        self.n_v = nv
        self.drop_ratio = drop
        self.ac_conv = activation_getter[ac_conv]
        self.ac_fc = activation_getter[ac_fc]

        # item embedding, 0 for padding
        self.item_embeddings = nn.Embedding(n_items + 1, d)

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

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization (同原始实现)
        self.item_embeddings.weight.data.normal_(
            0, 1.0 / self.item_embeddings.embedding_dim
        )

    def represent(self, seq_var):
        """
        卷积序列编码, 与原始实现 forward 的编码部分一致,
        返回序列表示 z: [B, d]。
        """
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)

        # vertical conv layer
        out_v = self.conv_v(item_embs)
        out_v = out_v.view(-1, self.fc1_dim_v)

        # horizontal conv layer
        out_hs = list()
        for conv in self.conv_h:
            conv_out = self.ac_conv(conv(item_embs).squeeze(3))
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)

        # fully-connected layer
        out = torch.cat([out_v, out_h], 1)
        out = self.dropout(out)
        z = self.ac_fc(self.fc1(out))
        return z

    def full_scores(self, seq_var):
        """
        一次前向计算所有物品得分(用于全量排序评估),
        共享物品嵌入点积打分。返回 [B, n_items],
        第 i 列对应物品 id i。
        """
        z = self.represent(seq_var)
        emb = self.item_embeddings.weight[1:]
        return z @ emb.t()

    def ce_loss(self, seq_var, target):
        """
        全词表 CrossEntropy(与 GAMER SeqModel 的 CE 损失一致)。
        target: [B] 或 [B, T], 物品 id(0 起始)。
        """
        z = self.represent(seq_var)
        emb = self.item_embeddings.weight[1:]
        logits = z @ emb.t()
        return F.cross_entropy(
            logits.reshape(-1, self.n_items), target.reshape(-1)
        )
