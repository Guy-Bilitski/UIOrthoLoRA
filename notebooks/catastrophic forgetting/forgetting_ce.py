"""
forgetting_ce.py  --  MiLoRA / Kalajdzievski "forgetting loss" for our saved adapters.

METRIC (exactly as MiLoRA Table 8 / MiLoRA sec.5.4, following Kalajdzievski 2024,
"Scaling laws for forgetting when fine-tuning LLMs", arXiv:2401.05605):

    "use cross-entropy as the metric for measuring forgetting. This is the usual
     next token prediction loss used when training LLMs, except that the target
     next token is replaced by the distribution predicted by the pre-trained base
     model.  ...evaluate on the WikiText-103 test dataset."

So per predicted position t (given left context x_<t) we compute the SOFT
cross-entropy between the pretrained base model's next-token distribution
(the TARGET / reference) and the fine-tuned model's next-token distribution:

    CE_t = H(p_base, p_ft) = - sum_v  p_base(v | x_<t) * log p_ft(v | x_<t)

FORGETTING = mean_t CE_t  over all predicted positions of the WikiText-103 test set.
This is the number directly comparable to MiLoRA Table 8 (LoRA 3.24, PiSSA 6.07,
MiLoRA 2.54).

We also report, for free, the two decompositions:
    H_base = mean_t [ -sum_v p_base(v) log p_base(v) ]        (base entropy; const across adapters)
    KL     = CE - H_base = mean_t KL( p_base(.|x<t) || p_ft(.|x<t) )   (Kalajdzievski forward-KL forgetting)
CE and KL differ only by the per-dataset constant H_base, so both give the SAME
ranking / correlation vs our magnitude axis. CE is the headline (matches MiLoRA units).

Direction note: the reference (first) argument is the BASE model (p_base), as MiLoRA
states ("target ... replaced by the distribution predicted by the pre-trained base
model"). That is forward KL(base||ft) == expected surprisal of ft under base targets.

BASE distribution is obtained via PeftModel.disable_adapter() on the SAME wrapped
model, so p_base uses exactly the pretrained W0 that p_ft = W0 + dW is built on. Our
residual-init methods (CorDA/MiLoRA/SC-LoRA/PiSSA) are saved as rank-2r W0-relative
adapters (residual_save.py), whose 0-step self-check guarantees "adapter disabled ==
original pretrained model", so disable_adapter() yields the genuine base for EVERY
method (plain LoRA included). Use --check_base to assert this against a fresh base.

DATA: WikiText-103 test.  NOTE the WikiText-2 and WikiText-103 *test* splits are
byte-identical (verified: 4358 rows / 1289979 chars); we load whichever config is
cached. Text is concatenated ("\n\n".join non-empty lines) and chunked into
non-overlapping blocks of length --max_length; CE is measured at every predicted
position (1..L-1) of every full block.

USAGE
  # single run, small validation slice, pick an idle GPU:
  CUDA_VISIBLE_DEVICES=5 python forgetting_ce.py --runs frm_lora_lr3e4_c256_s42 \
        --max_length 1024 --max_blocks 40 --batch_size 2

  # batch many runs (base model + tokenized data loaded ONCE, adapters hot-swapped):
  CUDA_VISIBLE_DEVICES=5 python forgetting_ce.py --runs run1,run2,run3 --max_blocks 0

  # full test set: --max_blocks 0
Writes results/<run>/forgetting.json and appends results/forgetting.jsonl.
"""
import os
import json
import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))


def load_wikitext_test_tokens(tokenizer, max_length, max_blocks, cache_dir):
    """Return a LongTensor [n_blocks, max_length] of WikiText-103 test token ids.

    WikiText-103 test == WikiText-2 test (identical). Load 103 if cached else 2."""
    from datasets import load_dataset
    ds = None
    last = None
    for cfg in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        try:
            ds = load_dataset("Salesforce/wikitext", cfg, split="test", cache_dir=cache_dir)
            print(f"[data] loaded WikiText test via config {cfg} ({len(ds)} rows)", flush=True)
            break
        except Exception as e:  # pragma: no cover
            last = e
    if ds is None:
        raise RuntimeError(f"Could not load WikiText test split: {last}")
    text = "\n\n".join(line for line in ds["text"] if line.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_full = ids.numel() // max_length
    ids = ids[: n_full * max_length].view(n_full, max_length)
    if max_blocks and max_blocks > 0:
        ids = ids[:max_blocks]
    print(f"[data] {ids.shape[0]} blocks x {max_length} tokens "
          f"= {ids.numel():,} tokens (~{ids.shape[0]*(max_length-1):,} predicted positions)", flush=True)
    return ids


@torch.no_grad()
def forgetting_for_model(model, blocks, batch_size, device, dtype=torch.bfloat16):
    """CE(p_base, p_ft), H_base and KL averaged over all predicted positions.

    p_base := model with adapter DISABLED (pretrained W0);  p_ft := adapter enabled."""
    tot_ce = torch.zeros((), dtype=torch.float64, device=device)
    tot_h = torch.zeros((), dtype=torch.float64, device=device)
    tot_n = 0
    for i in range(0, blocks.shape[0], batch_size):
        ids = blocks[i:i + batch_size].to(device)
        # fine-tuned (adapter ON) logits, then base (adapter OFF) logits
        ft_logits = model(input_ids=ids).logits[:, :-1, :].float()
        with model.disable_adapter():
            base_logits = model(input_ids=ids).logits[:, :-1, :].float()
        logp_base = F.log_softmax(base_logits, dim=-1)
        p_base = logp_base.exp()
        logp_ft = F.log_softmax(ft_logits, dim=-1)
        ce = -(p_base * logp_ft).sum(-1)          # [B, L-1] soft cross-entropy H(p_base,p_ft)
        h = -(p_base * logp_base).sum(-1)         # [B, L-1] base entropy H(p_base)
        tot_ce += ce.double().sum()
        tot_h += h.double().sum()
        tot_n += ce.numel()
        del ft_logits, base_logits, logp_base, p_base, logp_ft, ce, h
    ce_m = (tot_ce / tot_n).item()
    h_m = (tot_h / tot_n).item()
    return {"forgetting_ce": ce_m, "base_entropy": h_m, "forgetting_kl": ce_m - h_m,
            "n_positions": tot_n}


@torch.no_grad()
def check_base_matches_fresh(model, base_fresh, blocks, device, n=1):
    """Assert disable_adapter() logits == fresh pretrained-base logits (max |diff|)."""
    ids = blocks[:n].to(device)
    with model.disable_adapter():
        a = model(input_ids=ids).logits.float()
    b = base_fresh(input_ids=ids).logits.float()
    d = (a - b).abs().max().item()
    print(f"[check_base] max|disable_adapter - fresh_base| = {d:.3e}", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated run names (dir basenames under --adapters_root)")
    ap.add_argument("--adapters_root", default="/scratch/cf_models")
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--max_length", type=int, default=1024,
                    help="block/context length for WikiText chunking (default 1024)")
    ap.add_argument("--max_blocks", type=int, default=0,
                    help="0 = full test set; else first N blocks (validation slice)")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", "/scratch/hf_cache"))
    ap.add_argument("--check_base", action="store_true",
                    help="verify disable_adapter()==fresh base on the FIRST run (correctness gate)")
    ap.add_argument("--no_write", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[base] loading {args.base_model} ({dtype}) on {device}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device).eval()
    blocks = load_wikitext_test_tokens(tokenizer, args.max_length, args.max_blocks,
                                       os.path.join(args.hf_home, "datasets"))

    fresh_for_check = None
    if args.check_base:
        fresh_for_check = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=dtype).to(device).eval()

    model = None
    prev_adapter_name = None
    results = []
    for k, run in enumerate(runs):
        adir = os.path.join(args.adapters_root, run)
        if not os.path.exists(os.path.join(adir, "adapter_model.safetensors")):
            print(f"[skip] {run}: no adapter_model.safetensors", flush=True)
            continue
        aname = f"a{k}"
        t0 = time.time()
        if model is None:
            model = PeftModel.from_pretrained(base, adir, adapter_name=aname).eval()
        else:
            model.load_adapter(adir, adapter_name=aname)
            model.set_adapter(aname)
            if prev_adapter_name is not None:
                try:
                    model.delete_adapter(prev_adapter_name)
                except Exception:
                    pass
        prev_adapter_name = aname

        if args.check_base and fresh_for_check is not None and k == 0:
            d = check_base_matches_fresh(model, fresh_for_check, blocks, device)
            assert d < 1e-2, f"disable_adapter base mismatch {d:.3e} -- residual bookkeeping wrong"

        r = forgetting_for_model(model, blocks, args.batch_size, device, dtype)
        r["run_name"] = run
        r["max_length"] = args.max_length
        r["n_blocks"] = int(blocks.shape[0])
        r["wall_s"] = round(time.time() - t0, 1)
        try:
            r["method"] = json.load(open(os.path.join(adir, "adapter_config.json"))).get("peft_type")
            r["adapter_r"] = json.load(open(os.path.join(adir, "adapter_config.json"))).get("r")
        except Exception:
            pass
        # attach our magnitude axis if available
        sp = os.path.join(HERE, "results", run, "summary.json")
        if os.path.exists(sp):
            try:
                r["fdelta"] = json.load(open(sp)).get("headline", {}).get("fdelta")
            except Exception:
                pass
        results.append(r)
        print(f"[{run}] forgetting_CE={r['forgetting_ce']:.4f}  KL={r['forgetting_kl']:.4f}  "
              f"H_base={r['base_entropy']:.4f}  fdelta={r.get('fdelta')}  ({r['wall_s']}s)", flush=True)

        if not args.no_write:
            outp = os.path.join(HERE, "results", run, "forgetting.json")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            json.dump(r, open(outp, "w"), indent=2)
            with open(os.path.join(HERE, "results", "forgetting.jsonl"), "a") as fh:
                fh.write(json.dumps(r) + "\n")

    print("\n==== SUMMARY (sorted by forgetting_CE) ====", flush=True)
    for r in sorted(results, key=lambda x: x["forgetting_ce"]):
        print(f"  {r['run_name']:40s} CE={r['forgetting_ce']:.4f}  KL={r['forgetting_kl']:.4f}  "
              f"fdelta={r.get('fdelta')}", flush=True)


if __name__ == "__main__":
    main()
