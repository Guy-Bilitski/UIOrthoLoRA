import os
os.environ.setdefault("HF_HOME","/scratch/hf_cache"); os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from datasets import load_dataset
for name,cfg,split in [("google-research-datasets/nq_open",None,"train"),
                       ("cais/mmlu","auxiliary_train","train")]:
    try:
        d=load_dataset(name,cfg,split=split) if cfg else load_dataset(name,split=split)
        print(f"cached {name} {cfg or ''} {split}: {len(d)} rows",flush=True)
    except Exception as e:
        print(f"FAIL {name} {cfg}: {type(e).__name__}: {str(e)[:150]}",flush=True)
print("CALIB_CACHE_DONE",flush=True)
