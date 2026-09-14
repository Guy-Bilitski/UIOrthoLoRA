"""Tiny randomly initialized CPU models only; no trained-model evidence."""
import copy
import io

import pytest
import torch
from torch import nn
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification

from notebooks.iclr.campaign.diagnostics import aggregate_layers, diagnose_layer, subspace_report
from notebooks.iclr.campaign.modeling import (
    FrozenMLMProbe, attention_modules, insert_spectral, parameter_inventory,
    set_full_finetuning, set_head_only,
)
from notebooks.iclr.campaign.spectral import SpectralConfig, SpectralLinear


torch.set_num_threads(2)


def tiny_models():
    torch.manual_seed(31415)
    cfg=RobertaConfig(vocab_size=31,hidden_size=12,num_hidden_layers=2,
                      num_attention_heads=3,intermediate_size=16,max_position_embeddings=32,
                      hidden_dropout_prob=0.,attention_probs_dropout_prob=0.,num_labels=2)
    original=RobertaForMaskedLM(cfg).eval()
    classifier=RobertaForSequenceClassification(cfg).eval()
    classifier.roberta.load_state_dict(original.roberta.state_dict())
    inputs={"input_ids":torch.tensor([[0,4,5,6,2],[0,7,8,9,2]]),
            "attention_mask":torch.ones(2,5,dtype=torch.long)}
    return original,classifier,inputs


def test_original_frozen_probe_and_full_ft_embeddings():
    original,model,inputs=tiny_models()
    probe=FrozenMLMProbe(original)
    frozen={k:v.clone() for k,v in probe.head.state_dict().items()}
    with torch.no_grad():
        expected=original(**inputs).logits
        torch.testing.assert_close(probe.logits(model.roberta,inputs),expected,rtol=0,atol=0)
        # Full FT can modify input embeddings; output decoder must remain original.
        model.roberta.embeddings.word_embeddings.weight.add_(.2)
    assert probe.head.decoder.weight.data_ptr()!=model.roberta.embeddings.word_embeddings.weight.data_ptr()
    for key,value in probe.head.state_dict().items():
        torch.testing.assert_close(value,frozen[key],rtol=0,atol=0)
    labels=torch.full_like(inputs["input_ids"],-100)
    labels[:,2]=inputs["input_ids"][:,2]
    masked={k:v.clone() for k,v in inputs.items()}
    masked["input_ids"][:,2]=3
    result=probe.evaluate(model.roberta,masked,labels)
    assert result["masked_token_count"]==[1,1]
    assert result["masked_token_cross_entropy"]>0
    assert not any(p.requires_grad for p in probe.parameters())
    with pytest.raises(ValueError,match="masked token"):
        probe.evaluate(model.roberta,inputs,torch.full_like(labels,-100))


def test_paired_insertion_heads_fullft_and_cpu_reload():
    original,base,inputs=tiny_models()
    adapted,frozen,headonly,fullft=(copy.deepcopy(base) for _ in range(4))
    cfg=SpectralConfig(tail_size=4,initial_scaler=.1,initial_coefficient=.1)
    layers=insert_spectral(adapted,cfg)
    insert_spectral(frozen,cfg,freeze_adapter=True)
    set_head_only(headonly)
    set_full_finetuning(fullft)
    assert len(layers)==8
    torch.testing.assert_close(adapted(**inputs).logits,frozen(**inputs).logits,rtol=0,atol=0)
    torch.testing.assert_close(base(**inputs).logits,headonly(**inputs).logits,rtol=0,atol=0)
    for model in (headonly,frozen):
        assert all(n.startswith("classifier.") for n,p in model.named_parameters() if p.requires_grad)
    assert all(p.requires_grad for p in fullft.parameters())
    assert fullft.roberta.embeddings.word_embeddings.weight.requires_grad
    assert fullft.roberta.encoder.layer[0].intermediate.dense.weight.requires_grad
    assert parameter_inventory(adapted)["trainable_without_head"]==8*(4+2*12)
    # One CPU backward is an integration check, not a task-training smoke run.
    adapted.train()
    loss=adapted(**inputs,labels=torch.tensor([0,1])).loss
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapted.parameters())
    assert sum(m.h.grad.abs().sum().item() for m in layers.values())>0
    adapted.eval()
    expected=adapted(**inputs).logits.detach()
    buf=io.BytesIO()
    torch.save(adapted.state_dict(),buf)
    restored=copy.deepcopy(base)
    insert_spectral(restored,cfg)
    buf.seek(0)
    restored.load_state_dict(torch.load(buf,weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(**inputs).logits,expected,rtol=0,atol=0)
    for module in layers.values():
        module.merge()
    torch.testing.assert_close(adapted(**inputs).logits,expected,rtol=2e-5,atol=1e-7)
    for module in layers.values():
        module.unmerge()


@pytest.mark.parametrize("leading,scalers",[(True,True),(False,False)])
def test_diagnostics_original_frame_and_bounds(leading,scalers):
    torch.manual_seed(17)
    m=SpectralLinear(nn.Linear(6,8,dtype=torch.float64),
                     SpectralConfig(tail_size=2,leading_identity=leading,use_scalers=scalers))
    report0=diagnose_layer(m,[1,3],[(42,17),(123,2021)])
    assert report0["learned_since_insertion"]["energy"]==0
    assert report0["learned_since_insertion"]["p_cross"] is None
    with torch.no_grad():
        m.h.add_(.01)
        if scalers:
            m.e.add_(torch.linspace(-.03,.02,8))
            m.d.add_(torch.linspace(.01,-.02,6))
    report=diagnose_layer(m,[1,3])
    assert report["learned_since_insertion"]["energy"]>0
    assert report["no_crossing_sufficient_condition"]["applicable"]==(not leading and not scalers)
    for key,actual in report["bounds"]["operator_norms"].items():
        assert actual<=report["bounds"]["right_hand_sides"][key]+1e-12
    for record in report0["orientation_nulls"]:
        assert record["singular_value_error"]<1e-12
    agg=aggregate_layers({"a":report,"b":report},{"a":m.w_pre.square().sum().item(),"b":m.w_pre.square().sum().item()})
    assert sum(agg["pooled_fractions"].values())==pytest.approx(1)
    assert agg["pooled_relative_frobenius"]==pytest.approx(report["total"]["relative_frobenius"])
    assert agg["statistical_unit"]=="training_seed"
    a=torch.eye(6,dtype=torch.float64)[:,:4]
    b=torch.eye(6,dtype=torch.float64)[:,2:]
    drift=subspace_report(a,b)
    assert drift["chordal"]==pytest.approx(1)
    assert drift["mean_squared_sine_all_k"]==pytest.approx(.5)


@pytest.mark.parametrize("scalers,leading",[(True,True),(True,False),(False,True),(False,False)])
def test_delivered_legacy_unrotated_correspondence(scalers,leading):
    # Import the exact delivered package using PYTHONPATH=<isolated repo>/src.
    from peft.tuners.uiortholora.layer import Linear as LegacyLinear
    torch.manual_seed(42)
    base=nn.Linear(8,8,dtype=torch.float32)
    legacy=LegacyLinear(copy.deepcopy(base),"default",num_svalues_to_adapt=3,
                        num_svectors_to_adapt=0,scaling_factor=1.,enforce_sv_positive=False,
                        initial_scaler=.1,initial_sigma=.01,use_de=scalers,drop_major=not leading)
    new=SpectralLinear(copy.deepcopy(base),SpectralConfig(tail_size=3,use_scalers=scalers,
                       leading_identity=leading,initial_scaler=.1,initial_coefficient=.01))
    x=torch.randn(4,8)
    torch.testing.assert_close(legacy.get_delta_weight("default"),new.delta_total(),rtol=2e-5,atol=1e-7)
    torch.testing.assert_close(legacy(x),new(x),rtol=2e-5,atol=1e-7)
