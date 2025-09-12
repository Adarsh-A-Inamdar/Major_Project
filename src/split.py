# src/split.py
import os, random, shutil
from pathlib import Path

random.seed(42)
PROC=Path("data/processed")
for cls in os.listdir(PROC):
    files=list((PROC/cls).glob("*.*"))
    random.shuffle(files)
    n=len(files); n_tr=int(0.7*n); n_va=int(0.15*n)
    for name, group in [("train",files[:n_tr]), ("val",files[n_tr:n_tr+n_va]), ("test",files[n_tr+n_va:])]:
        out=Path(f"data/{name}/{cls}"); out.mkdir(parents=True, exist_ok=True)
        for f in group: shutil.copy2(f, out/f.name)
