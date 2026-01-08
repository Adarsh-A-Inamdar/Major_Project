# src/preprocess.py
import cv2, os, glob
from pathlib import Path

RAW="data/raw"; 
PROC="data/processed"; 
SIZE=224
for cls in os.listdir(RAW):
    in_dir=Path(RAW)/cls; out_dir=Path(PROC)/cls; out_dir.mkdir(parents=True, exist_ok=True)
    for fp in glob.glob(str(in_dir/'*.jpg'))+glob.glob(str(in_dir/'*.png')):
        img=cv2.imread(fp)
        if img is None: continue
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(SIZE,SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir/Path(fp).name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
