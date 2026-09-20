import torch
from transformers import T5Config, T5ForConditionalGeneration

PAD_ID = 0
EOS_ID = 1
SPECIAL_TOKENS = 2


def tiger_vocab_size(sid_table):
    return SPECIAL_TOKENS + sid_table.num_levels * sid_table.codebook_size


def build_tiger(
    sid_table,
    d_model=128,
    d_ff=512,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
):
    config = T5Config(
        vocab_size=tiger_vocab_size(sid_table),
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        num_decoder_layers=num_layers,
        dropout_rate=dropout,
        feed_forward_proj="relu",
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
        decoder_start_token_id=PAD_ID,
    )
    return T5ForConditionalGeneration(config)


def encode_history(hist_sids, sid_table, max_len=30):
    tokens = []
    for sid in hist_sids:
        tokens.extend(sid_table.sid_to_token_ids(sid))
    return tokens[:max_len]


def encode_target(tgt_sid, sid_table):
    return sid_table.sid_to_token_ids(tgt_sid) + [EOS_ID]


def make_prefix_allowed_fn(sid_table, offset=SPECIAL_TOKENS):
    def fn(batch_id, input_ids):
        generated = [
            t for t in input_ids.tolist() if t >= offset
        ]
        level = len(generated)
        if level >= sid_table.num_levels:
            return [EOS_ID]
        prefix = []
        for t in generated:
            code = (t - offset) % sid_table.codebook_size
            prefix.append(code)
        allowed = sid_table.allowed_codes(tuple(prefix))
        return [
            sid_table.sid_token_id(level, c, offset) for c in allowed
        ]

    return fn


def decode_beam_to_sid(token_ids, codebook_size=256, offset=SPECIAL_TOKENS):
    codes = [
        (t - offset) % codebook_size for t in token_ids if t >= offset
    ]
    return tuple(codes)


@torch.no_grad()
def beam_search_items(
    model,
    input_ids,
    attention_mask,
    sid_table,
    num_beams=20,
    top_k=10,
    batch_size=64,
    device="cpu",
    offset=SPECIAL_TOKENS,
):
    model.eval()
    prefix_fn = make_prefix_allowed_fn(sid_table, offset)
    results = []
    for start in range(0, input_ids.size(0), batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = attention_mask[start : start + batch_size].to(device)
        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=sid_table.num_levels + 1,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            prefix_allowed_tokens_fn=prefix_fn,
            return_dict_in_generate=True,
            output_scores=True,
        )
        seqs = out.sequences
        scores = out.sequences_scores
        n_rows = ids.size(0)
        for r in range(n_rows):
            row_scores = scores[r * num_beams : (r + 1) * num_beams]
            order = torch.argsort(row_scores, descending=True).tolist()
            items = []
            seen = set()
            for b in order:
                i = r * num_beams + b
                beam_tokens = seqs[i].tolist()
                sid = decode_beam_to_sid(
                    beam_tokens, sid_table.codebook_size, offset
                )
                for item in sid_table.sid2items.get(sid, []):
                    if item not in seen:
                        seen.add(item)
                        items.append(item)
                        if len(items) >= top_k:
                            break
                if len(items) >= top_k:
                    break
            results.append(items)
    return results
