import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(keys)
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(
            torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output
        )
        diag_vals = torch.ones_like(matmul_output[0, :, :])
        tril = torch.tril(diag_vals)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(
            torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1
        )
        matmul_output_sm = self.softmax(matmul_output_m2)
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])
        matmul_output_qm = matmul_output_sm * query_mask
        matmul_output_dropout = self.dropout(matmul_output_qm)
        output_ws = torch.bmm(matmul_output_dropout, V_)
        output = torch.cat(
            torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2
        )
        output_res = output + queries
        return output_res


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class SASRecMiniOneRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, num_heads=1):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1, embedding_dim=hidden_size
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size, embedding_dim=hidden_size
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_size, hidden_size, dropout
        )
        self.s_fc = nn.Linear(hidden_size, item_num)

    def forward(self, states, len_states):
        device = states.device
        inputs_emb = self.item_embeddings(states)
        inputs_emb = inputs_emb + self.positional_embeddings(
            torch.arange(self.state_size, device=device)
        )
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        seq = seq * mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out = ff_out * mask
        ff_out = self.ln_3(ff_out)
        indices = (len_states - 1).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze(1)
        return supervised_output
