import ast
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from src.data.sid import SidTable

_SID_RE = re.compile(r"<[abc]_(\d+)>")


@dataclass
class Split:
    users: list = field(default_factory=list)
    histories: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    hist_sids: list = field(default_factory=list)
    tgt_sids: list = field(default_factory=list)

    def __len__(self):
        return len(self.targets)


@dataclass
class DatasetBundle:
    category: str
    sid_table: SidTable
    train: Split
    valid: Split
    test: Split

    @property
    def n_items(self):
        return self.sid_table.n_items


def _parse_sid_string(s):
    codes = [int(m) for m in _SID_RE.findall(s)]
    assert len(codes) == 3, s
    return tuple(codes)


def _load_split(csv_path):
    df = pd.read_csv(csv_path)
    split = Split()
    split.users = df["user_id"].tolist()
    split.histories = [ast.literal_eval(s) for s in df["history_item_id"]]
    split.targets = [int(t) for t in df["item_id"]]
    split.hist_sids = [
        [_parse_sid_string(x) for x in ast.literal_eval(s)]
        for s in df["history_item_sid"]
    ]
    split.tgt_sids = [_parse_sid_string(s) for s in df["item_sid"]]
    return split


def load_bundle(data_root, category):
    fname = f"{category}_5_2016-10-2018-11.csv"
    sid_table = SidTable.from_index_json(
        os.path.join(data_root, "index", f"{category}.index.json")
    )
    splits = {}
    for name in ("train", "valid", "test"):
        splits[name] = _load_split(os.path.join(data_root, name, fname))
    return DatasetBundle(
        category=category,
        sid_table=sid_table,
        train=splits["train"],
        valid=splits["valid"],
        test=splits["test"],
    )
