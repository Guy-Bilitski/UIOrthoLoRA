"""CE-drift (forgetting) for a DeepSeek-V4-Flash adapter, sharded across 8 GPUs.

Same protocol/output as forgetting_ce.py: soft CE(p_base, p_ft) + KL on WikiText-103 test,
where p_base = the SAME model with the adapter disabled (disable_adapter()). Differences:
  - base loaded dequant->bf16, device_map="auto"; adapter injected onto it.
  - accumulation is device-agnostic (logits live on the last pipeline stage) and the per-position
    CE is computed in vocab CHUNKS so the 129,280-way softmax never materializes in full.
Writes results/<run>/forgetting.json and appends results/forgetting_deepseek.jsonl (7B schema).

Usage:
  HF_HOME=/scratch/hf_cache HF_HUB_OFFLINE=1 python3 scripts/deepseek/ce_deepseek.py \
    --adapter /scratch/cf_models/dsv4_lora_r16_lr3e4_s42 --max_length 1024 --max_blocks 40
"""
import os, sys, json, time, argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import run_lib
from forgetting_ce import load_wikitext_test_tokens

MODEL = "deepseek-ai/DeepSeek-V4-Flash"


@torch.no_grad()
def _ce_kl_chunked(base_logits, ft_logits, vocab_chunk=16384):
    """CE(p_base,p_ft), H(p_base) summed over positions, computing the softmax normalizers and
    the cross terms in vocab chunks to bound memory. logits: [P, V] (already flattened positions)."""
    P, V = base_logits.shape
    # log-sum-exp normalizers per position (streamed over vocab)
    def lse(x):
        m = torch.full((P,), float("-inf"), device=x.device)
        for s in range(0, V, vocab_chunk):
            m = torch.maximum(m, x[:, s:s + vocab_chunk].max(dim=1).values)
        acc = torch.zeros(P, dtype=torch.float64, device=x.device)
        for s in range(0, V, vocab_chunk):
            acc += (x[:, s:s + vocab_chunk] - m.unsqueeze(1)).exp().sum(dim=1).double()
        return m.double() + acc.log()
    lse_b = lse(base_logits)
    lse_f = lse(ft_logits)
    tot_ce = torch.zeros(P, dtype=torch.float64, device=base_logits.device)
    tot_h = torch.zeros(P, dtype=torch.float64, device=base_logits.device)
    for s in range(0, V, vocab_chunk):
        b = base_logits[:, s:s + vocab_chunk].double()
        f = ft_logits[:, s:s + vocab_chunk].double()
        logp_b = b - lse_b.unsqueeze(1)
        p_b = logp_b.exp()
        logp_f = f - lse_f.unsqueeze(1)
        tot_ce += -(p_b * logp_f).sum(dim=1)
        tot_h += -(p_b * logp_b).sum(dim=1)
    return tot_ce.sum().item(), tot_h.sum().item(), P


@torch.no_grad()
def forgetting_sharded(model, blocks, in_dev, vocab_chunk):
    tot_ce = tot_h = 0.0
    tot_n = 0
    for i in range(blocks.shape[0]):
        ids = blocks[i:i + 1].to(in_dev)
        ft = model(input_ids=ids).logits[:, :-1, :].float().squeeze(0)   # [L-1, V] on last stage
        with model.disable_adapter():
            base = model(input_ids=ids).logits[:, :-1, :].float().squeeze(0)
        ce, h, n = _ce_kl_chunked(base, ft, vocab_chunk)
        tot_ce += ce; tot_h += h; tot_n += n
        del ft, base
    ce_m = tot_ce / tot_n
    h_m = tot_h / tot_n
    return {"forgetting_ce": ce_m, "base_entropy": h_m, "forgetting_kl": ce_m - h_m, "n_positions": tot_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_blocks", type=int, default=40, help="0=full WikiText test")
    ap.add_argument("--vocab_chunk", type=int, default=16384)
    ap.add_argument("--out", default="results/forgetting_deepseek.jsonl")
    args = ap.parse_args()
    run_name = args.run_name or os.path.basename(os.path.normpath(args.adapter))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    from transformers import FineGrainedFP8Config
    print(f"[load] {MODEL} dequant->bf16 device_map=auto ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True),
        low_cpu_mem_usage=True, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    in_dev = model.get_input_embeddings().weight.device

    blocks = load_wikitext_test_tokens(tokenizer, args.max_length, args.max_blocks,
                                       cache_dir=os.environ.get("HF_HOME"))
    t0 = time.time()
    r = forgetting_sharded(model, blocks, in_dev, args.vocab_chunk)
    r["wall_s"] = round(time.time() - t0, 1)
    r["run_name"] = run_name
    r["max_length"] = args.max_length
    r["n_blocks"] = int(blocks.shape[0])
    try:
        r["method"] = json.load(open(os.path.join(args.adapter, "adapter_config.json"))).get("peft_type", "unknown")
        r["adapter_r"] = json.load(open(os.path.join(args.adapter, "adapter_config.json"))).get("r")
    except Exception:
        pass
    sp = os.path.join(HERE, "results", run_name, "summary.json")
    if os.path.exists(sp):
        r["fdelta"] = (json.load(open(sp)).get("headline") or {}).get("fdelta")
    run_lib.write_json(os.path.join(HERE, "results", run_name, "forgetting.json"), r)
    with open(os.path.join(HERE, args.out), "a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"==== {run_name} CE={r['forgetting_ce']:.4f} KL={r['forgetting_kl']:.4f} "
          f"H_base={r['base_entropy']:.4f} n={r['n_positions']} ({r['wall_s']}s) ====", flush=True)


if __name__ == "__main__":
    main()
