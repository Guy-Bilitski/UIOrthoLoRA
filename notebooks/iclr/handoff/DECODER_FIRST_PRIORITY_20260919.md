# Author priority update — modern decoder first

The author's latest instruction is: "the most important thing right now is to
have a modern decoder in the results ... Much before another controls."

This supersedes the experiment order in the assessment, preparation handoff,
coding prompt and coding-session schedule. Keep their scientific definitions,
validation requirements and immutable evidence. The budget remains two GPUs.

1. **First new GPU work: the modern decoder pilot.** Use the prepared
   `notebooks/iclr/decoder_pilot/` package and `DECODER_PILOT_DESIGN_20260919.md`.
   Qwen2.5-1.5B-Instruct/GSM8K remains the prepared candidate. Resolve the specific
   recipe and assigned device IDs, run the memory/throughput/learning pilot,
   then prioritize its tuning, calibration and confirmation runs. GPU costs in
   the design are estimates until measured by that pilot.
2. **Protect a complete, interpretable decoder comparison.** The proposed
   UNREG/MIX/NORM/LoRA/PiSSA comparison and three seeds remain the target for
   discussion. Preserve the comparisons needed to interpret the decoder result;
   do not put additional RoBERTa controls ahead of it. Freeze any budget-driven
   scope change before confirmation outcomes. If time is tight, defer CENTER
   and optional extensions before reducing decoder replication.
3. **CENTER and module-allocation controls are deferred.** Do not start their
   calibration or confirmations ahead of the decoder. The prepared code remains
   available for later use if the decoder result and remaining time justify it.
   Do not automatically replace an expensive decoder recipe with more encoder
   experiments; first consider a narrower, still informative decoder study.
4. **Existing-checkpoint work is already delivered.** See
   `EXPERIMENT_PREPARATION_STATUS_20260919.md`,
   `data/temperature_analysis_20260916.json`, `data/band_held_aside_20260917/`
   and `data/run_records_20260919/`. Do not repeat inference or describe these
   exports as missing. CPU verification and writing may continue without
   delaying the decoder or consuming its assigned GPU budget.
5. **Writing follows the new evidence.** Integrate the completed calibration
   and band evaluations and then the decoder result with accurate scope. The
   temperature export reports that most of the raw MIX--NORM NLL gap disappears
   after inner-fitted temperature scaling; this requires an interpretation
   update, not a claim of equivalent models. The band export also discloses
   training-time locked-split scoring. Preserve that provenance when integrating
   it. The current paper has not yet integrated these new scientific results.

The execution order is decoder pilot -> decoder tuning/calibration -> decoder
confirmations -> optional controls if time remains. Reserve the last 48 hours
for analysis/writing. This priority update does not invent device assignments,
change frozen protocols or authorize taking another user's GPU. It does not
choose every open decoder hyperparameter or launch a job from this paper session.
