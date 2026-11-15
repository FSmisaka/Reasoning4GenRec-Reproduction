import glob
import json
import os

DS, CAT = {"games": ("Games", "Video_Games"),
           "office": ("Office", "Office_Products"),
           "industrial": ("Industrial", "Industrial_and_Scientific")}[os.environ.get("DATASET", "games")]
legal = {l.split("\t")[0].strip() for l in open(
    f"data/raw/Amazon/info/{CAT}_5_2016-10-2018-11.txt")}
files = sorted(glob.glob(f"results/sidreasoner/{DS}/think_*_temp/*_seed*.json"))
preds = [sid for f in files for s in json.load(open(f)) for sid in s["predict"]]
bad = sorted({sid for sid in preds if sid not in legal})
print(f"files={len(files)} sids={len(preds)} legal={len(preds) - len([s for s in preds if s not in legal])} "
      f"({(len(preds) - len([s for s in preds if s not in legal])) / len(preds):.2%})")
print("illegal examples:", bad[:10])
assert not bad
