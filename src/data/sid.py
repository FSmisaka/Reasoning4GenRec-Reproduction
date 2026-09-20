import json
import re
from dataclasses import dataclass, field


@dataclass
class SidTable:
    item2sid: dict
    sid2items: dict
    codebook_size: int = 256
    num_levels: int = 3
    trie: dict = field(default_factory=dict)

    @property
    def n_items(self):
        return len(self.item2sid)

    @classmethod
    def from_index_json(cls, path, codebook_size=256):
        raw = json.load(open(path))
        item2sid = {}
        for k, toks in raw.items():
            codes = tuple(
                int(re.fullmatch(r"<([abc])_(\d+)>", t).group(2)) for t in toks
            )
            assert len(codes) == 3
            item2sid[int(k)] = codes
        sid2items = {}
        for item in sorted(item2sid):
            sid2items.setdefault(item2sid[item], []).append(item)
        trie = {}
        for sid in sid2items:
            node = trie
            for level, code in enumerate(sid):
                node = node.setdefault(code, {})
        return cls(item2sid, sid2items, codebook_size, 3, trie)

    def allowed_codes(self, prefix):
        node = self.trie
        for code in prefix:
            node = node.get(code)
            if node is None:
                return set()
        return set(node.keys())

    def sid_token_id(self, level, code, offset=2):
        return offset + level * self.codebook_size + code

    def sid_to_token_ids(self, sid, offset=2):
        return [
            self.sid_token_id(l, c, offset) for l, c in enumerate(sid)
        ]
