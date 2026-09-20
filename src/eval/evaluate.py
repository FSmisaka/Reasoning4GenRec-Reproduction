import torch

from src.eval.metrics import rank_metrics, rank_of
from src.models.tiger.model import beam_search_items, encode_history


def _pad_histories(histories, max_len, pad=0):
    seq = torch.full((len(histories), max_len), pad, dtype=torch.long)
    lens = torch.zeros(len(histories), dtype=torch.long)
    for i, h in enumerate(histories):
        h = h[-max_len:]
        seq[i, max_len - len(h) :] = torch.tensor(h, dtype=torch.long) + 1
        lens[i] = len(h)
    return seq


@torch.no_grad()
def evaluate_sasrec(
    model, split, device, k_list=(5, 10), batch_size=256, max_len=10
):
    model.eval()
    ranks = []
    for start in range(0, len(split), batch_size):
        batch = split.targets[start : start + batch_size]
        hist = split.histories[start : start + batch_size]
        seq = _pad_histories(hist, max_len).to(device)
        scores = model.full_scores(seq)
        _, topk = torch.topk(scores, max(k_list), dim=1)
        topk = topk.cpu().tolist()
        for i, tgt in enumerate(batch):
            ranks.append(rank_of(tgt, topk[i]))
    return rank_metrics(ranks, k_list)


@torch.no_grad()
def evaluate_tiger(
    model,
    split,
    sid_table,
    device,
    k_list=(5, 10),
    num_beams=20,
    batch_size=64,
    max_len=30,
):
    max_tokens = max_len * sid_table.num_levels
    histories = split.hist_sids
    tokenized = [encode_history(h, sid_table, max_tokens) for h in histories]
    L = max(len(t) for t in tokenized)
    input_ids = torch.full((len(tokenized), L), 0, dtype=torch.long)
    attention_mask = torch.zeros(len(tokenized), L, dtype=torch.long)
    for i, toks in enumerate(tokenized):
        input_ids[i, : len(toks)] = torch.tensor(toks)
        attention_mask[i, : len(toks)] = 1
    ranked = beam_search_items(
        model,
        input_ids,
        attention_mask,
        sid_table,
        num_beams=num_beams,
        top_k=max(k_list),
        batch_size=batch_size,
        device=device,
    )
    ranks = [
        rank_of(tgt, ranked[i]) for i, tgt in enumerate(split.targets)
    ]
    return rank_metrics(ranks, k_list)
