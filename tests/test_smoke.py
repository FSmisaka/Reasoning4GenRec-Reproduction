import sys

import torch

sys.path.insert(0, ".")

from src.data.bundle import load_bundle
from src.data.sid import SidTable
from src.eval.evaluate import evaluate_sasrec
from src.models.sasrec.model import SASRec
from src.models.tiger.model import (
    build_tiger,
    encode_history,
    encode_target,
    make_prefix_allowed_fn,
    beam_search_items,
)


def main():
    bundle = load_bundle("data/raw/Amazon", "Video_Games")
    sid = bundle.sid_table
    assert bundle.n_items == 3858
    assert len(bundle.train) == 49133 and len(bundle.valid) == 6142

    allowed0 = sid.allowed_codes(())
    assert len(allowed0) > 0
    first = next(iter(allowed0))
    allowed1 = sid.allowed_codes((first,))
    assert len(allowed1) > 0
    probe = sid.item2sid[0]
    assert sid.allowed_codes(probe[:2]) == {probe[2]}

    from src.data.bundle import Split

    mini = Split(
        histories=bundle.train.histories[:8],
        targets=bundle.train.targets[:8],
        hist_sids=bundle.train.hist_sids[:8],
        tgt_sids=bundle.train.tgt_sids[:8],
    )

    device = "cpu"
    model = SASRec(n_items=bundle.n_items)
    seq = torch.zeros(8, 10, dtype=torch.long)
    labels = torch.zeros(8, 10, dtype=torch.long)
    exclude = torch.zeros(8, bundle.n_items + 1, dtype=torch.bool)
    for i in range(8):
        hist = mini.histories[i]
        tgt = mini.targets[i]
        full = [x + 1 for x in hist] + [tgt + 1]
        inp, lab = full[:-1], full[1:]
        seq[i, 10 - len(inp) :] = torch.tensor(inp)
        labels[i, 10 - len(lab) :] = torch.tensor(lab)
        exclude[i, sorted(set(full) | {0})] = True
    loss = model.bce_loss(seq, labels, exclude)
    assert torch.isfinite(loss), loss.item()
    loss.backward()
    loss_ce = model.ce_loss(seq, labels)
    assert torch.isfinite(loss_ce)
    m = evaluate_sasrec(model, mini, device, batch_size=4)
    assert set(m) == {"Recall@5", "NDCG@5", "Recall@10", "NDCG@10"}
    print("SASRec smoke ok, loss:", round(loss.item(), 4))

    tiger = build_tiger(sid).to(device)
    enc_list = [encode_history(h, sid) for h in mini.hist_sids]
    L = max(len(t) for t in enc_list)
    enc = torch.zeros(len(enc_list), L, dtype=torch.long)
    for i, t in enumerate(enc_list):
        enc[i, : len(t)] = torch.tensor(t)
    mask = (enc != 0).long()
    dec = torch.tensor([encode_target(t, sid) for t in mini.tgt_sids])
    mask = (enc != 0).long()
    out = tiger(input_ids=enc, attention_mask=mask, labels=dec)
    assert torch.isfinite(out.loss), out.loss.item()
    out.loss.backward()
    print("TIGER smoke ok, loss:", round(out.loss.item(), 4))

    ranked = beam_search_items(
        tiger, enc, mask, sid, num_beams=8, top_k=10, batch_size=4,
        device=device,
    )
    assert len(ranked) == 8
    for items in ranked:
        assert len(items) >= 1
        assert all(0 <= i < bundle.n_items for i in items)
        assert len(items) <= 10
    print("TIGER beam smoke ok, example:", ranked[0][:5])
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
