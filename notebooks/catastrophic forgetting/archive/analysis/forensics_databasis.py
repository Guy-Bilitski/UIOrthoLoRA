"""
DATA-BASIS forensic (the basis-reveal test): does the DATA/activation-covariance basis predict
retention while the static WEIGHT-SVD basis does not? (See handoff/06_INSIGHTS.md ★ + 07_RELATED_WORK.)

For each updated matrix we capture the input-activation covariance C_X = E[x xᵀ] over a calibration
set, then compare how much the trained ΔW overlaps the TOP directions of two bases:
  - WEIGHT basis:  V_w = top-r right singular vectors of W0            (OPLoRA / CLoRA / UIOrtho basis)
  - DATA   basis:  V_d = top-r eigenvectors of C_X                     (the directions inputs occupy)
  - CorDA  basis:  V_c = top-r right singular vectors of W0·C_X^{1/2}  (CorDA's context basis)
metrics (fraction of ΔW's input-energy in the top subspace, lower = stays off important dirs):
  w_inTop  = ||ΔW V_w||_F² / ||ΔW||_F²
  d_inTop  = ||ΔW V_d||_F² / ||ΔW||_F²
  c_inTop  = ||ΔW V_c||_F² / ||ΔW||_F²
plus data_resp = tr(ΔW C_X ΔWᵀ)/tr(C_X)  (data-weighted magnitude ≈ F-delta², the real disruption).

    python forensics_databasis.py --adapter /scratch/cf_models/clora_cs_k2048 --run_name clora_k2048 --rfrac 0.1
"""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
import run_lib, fdelta

HERE = run_lib.HERE


def load_retain_inputs(n, tok):
    """RETAINED-knowledge distribution (MMLU-Pro = our out-domain retention benchmark) — the directions
    forgetting is measured on. This is the CORRECT covariance for predicting forgetting (NOT the
    commonsense fine-tuning task, which fdelta.load_inputs returns = C_task)."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    prompts = []
    for d in ds.select(range(min(len(ds), n * 3))):
        opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(d["options"]))
        prompts.append(f"{d['question']}\n{opts}\nAnswer:")
        if len(prompts) >= n:
            break
    return prompts[:n]


@torch.no_grad()
def basis_metrics(W0, dW, CX, rfrac):
    W0=W0.float(); dW=dW.float(); CX=CX.float()
    inf=W0.shape[1]; r=max(1,int(round(rfrac*inf)))
    E=(dW*dW).sum().item()
    if E<=0: return None
    # weight basis: top-r right singular vectors of W0
    _,_,Vtw=torch.linalg.svd(W0,full_matrices=False); Vw=Vtw[:r].transpose(-1,-2)        # (in,r)
    # data basis: top-r eigenvectors of C_X (symmetric PSD)
    lam,Q=torch.linalg.eigh(CX)                                                            # ascending
    Vd=Q[:, -r:]                                                                           # (in,r) top-r
    CXh=(Q*lam.clamp_min(0).sqrt()) @ Q.transpose(-1,-2)                                    # C_X^{1/2}
    # CorDA basis: top-r right singular vectors of W0 C_X^{1/2}
    _,_,Vtc=torch.linalg.svd(W0@CXh,full_matrices=False); Vc=Vtc[:r].transpose(-1,-2)      # (in,r)
    def frac(V): return ((dW@V)**2).sum().item()/E
    data_resp=torch.trace(dW@CX@dW.transpose(-1,-2)).item()/ (torch.trace(CX).item()+1e-9)
    return dict(w_inTop=frac(Vw), d_inTop=frac(Vd), c_inTop=frac(Vc), data_resp=data_resp,
                dwF=E**0.5, r=r, inf=inf)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter",required=True); ap.add_argument("--run_name",default="")
    ap.add_argument("--n_inputs",type=int,default=64); ap.add_argument("--rfrac",type=float,default=0.1)
    ap.add_argument("--cov_source",default="retain",choices=["retain","task"],
                    help="retain=MMLU-Pro (out-domain; predicts FORGETTING — default); task=commonsense (C_task, for the zero-sum trade-off)")
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.base_model); tok.pad_token_id=0; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(args.base_model,dtype=torch.bfloat16,device_map="cuda:0")
    model=PeftModel.from_pretrained(model,args.adapter,device_map={"":0}); model.eval()
    # accumulate C_X per updated matrix via input pre-hooks
    cov={}; hooks=[]
    def mk(name):
        def h(mod,inp):
            x=inp[0].reshape(-1,inp[0].shape[-1]).float()
            cov[name]=cov.get(name,0.0)+ (x.transpose(-1,-2)@x)
        return h
    layers={}
    for n,m in model.named_modules():
        if isinstance(m,BaseTunerLayer) and hasattr(m,"get_delta_weight"):
            try: layers[n]=(m.get_delta_weight("default").detach(), m.get_base_layer().weight.detach())
            except: continue
            hooks.append(m.register_forward_pre_hook(mk(n)))
    if args.cov_source=="retain":
        try: prompts=load_retain_inputs(args.n_inputs,tok)
        except Exception as e:
            print(f"[databasis] retain (MMLU-Pro) load FAILED ({e}); falling back to task covariance",flush=True)
            args.cov_source="task_fallback"; prompts=fdelta.load_inputs(args.n_inputs,tok)
    else:
        prompts=fdelta.load_inputs(args.n_inputs,tok)
    print(f"[databasis] cov_source={args.cov_source} ({len(prompts)} prompts)",flush=True)
    for i in range(0,len(prompts),8):
        enc=tok(prompts[i:i+8],return_tensors="pt",padding=True,truncation=True,max_length=256).to("cuda:0")
        model(**enc)
    for h in hooks: h.remove()
    rows=[]
    for n,(dW,W0) in layers.items():
        if n not in cov: continue
        m=basis_metrics(W0,dW,cov[n],args.rfrac)
        if m: rows.append(m)
    tot=sum(r["dwF"]**2 for r in rows) or 1.0
    keys=[k for k in rows[0] if k not in("r","inf")]
    agg={k:round(sum(r[k]*r["dwF"]**2 for r in rows)/tot,5) for k in keys}
    agg["n_matrices"]=len(rows); agg["rfrac"]=args.rfrac
    rn=args.run_name or os.path.basename(os.path.normpath(args.adapter))
    agg["cov_source"]=args.cov_source
    run_lib.write_json(os.path.join(HERE,"results",f"databasis_{rn}_{args.cov_source}.json"),
                       {"run_name":rn,"adapter":args.adapter,"kind":"databasis","cov_source":args.cov_source,"agg":agg})
    print(f"[databasis] {rn} [cov={args.cov_source}]: WEIGHT-basis inTop={agg['w_inTop']:.3f}  "
          f"DATA-basis inTop={agg['d_inTop']:.3f}  CorDA-basis inTop={agg['c_inTop']:.3f}  "
          f"data_resp(||dW.C^1/2||_F^2 norm)={agg['data_resp']:.3f} (rfrac={args.rfrac})",flush=True)


if __name__=="__main__":
    main()
