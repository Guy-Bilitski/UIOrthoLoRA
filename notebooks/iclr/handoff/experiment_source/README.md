# Recorded experiment implementation

This snapshot contains the 46 Python files fingerprinted in the executed band
recipe manifests, including the adapter implementations, training engine,
checkpoint validation code and tests. It was retrieved from the recorded source
revision and checked byte for byte against the manifests' SHA256 fingerprints
and Git blob identities. All ten band-run revisions have identical campaign
trees. `data/band_source_audit.json` records this comparison.

The supplementary archive anonymizes the private experiment identifier in two
files. Its audit records both original and distributed hashes. No other source
changes are made. `requirements.lock.txt` comes from the same recorded revision;
it describes the server environment, not the lighter analysis environment.

Code is laid out as `notebooks.iclr.campaign` under this directory. The modules
`band.py`, `spectral.py`, `regularizers.py` and `engine.py` implement the main
constructions and optimization; `worker.py` orchestrates a run. The executed
recipes and initialization/training settings are in the evidence manifests.

This is an inspectable implementation snapshot, not a self-contained training
replay. Prepared dataset tensors, base-model weights, protocol admission files,
saved adapter checkpoints and original validation reports are not included.
The analysis verifier does not install the training environment, launch jobs or
rerun checkpoint validation. Its source checks establish correspondence between
the exported records and this snapshot; they do not independently authenticate
execution on the server.
