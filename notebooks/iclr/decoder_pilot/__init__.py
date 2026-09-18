"""Compact decoder pilot (Qwen2.5-1.5B-Instruct / GSM8K candidate) for the ICLR interaction study.

Isolated from the RoBERTa campaign runner: it reuses the tested spectral layer,
LoRA layer, regularizers and pretrained-frame block diagnostics from
``notebooks.iclr.campaign`` and adds decoder-specific data, PiSSA, training and
generation evaluation. Nothing here is a completed result; see
``handoff/DECODER_PILOT_DESIGN_20260919.md`` for the proposed protocol and the
decisions that remain open.
"""
