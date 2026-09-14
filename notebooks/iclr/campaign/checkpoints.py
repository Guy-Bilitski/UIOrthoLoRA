"""Durable reference plus compact restart states, never replacing old artifacts."""

import contextlib
import copy
import json
import os
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch

from .artifacts import sha256, write_json_new


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_torch_new(path, state):
    """A partial write never masquerades as a committed, verified checkpoint."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial_" + uuid4().hex)
    if path.exists():
        raise FileExistsError(path)
    with partial.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    # Hard-link creation is atomic and fails if the destination already exists.
    os.link(partial, path)
    sync_directory(path.parent)
    partial.unlink()  # Only our successfully committed temporary link.
    return sha256(path)


def capture_rng():
    numpy_state = np.random.get_state()
    state = dict(
        python=random.getstate(),
        torch_cpu=torch.get_rng_state(),
        numpy=dict(
            kind=numpy_state[0],
            keys=numpy_state[1].tolist(),
            pos=numpy_state[2],
            has_gauss=numpy_state[3],
            cached_gaussian=numpy_state[4],
        ),
        torch_cuda=None,
    )
    if torch.cuda.is_initialized():
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Checkpoint worker must expose exactly one assigned GPU")
        state["torch_cuda"] = torch.cuda.get_rng_state(0)
    return cpu_copy(state)


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    n = state["numpy"]
    np.random.set_state(
        (n["kind"], np.array(n["keys"], dtype=np.uint32), n["pos"], n["has_gauss"], n["cached_gaussian"])
    )
    if state["torch_cuda"] is not None:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Restoring CUDA RNG requires exactly one assigned visible GPU")
        torch.cuda.set_rng_state(state["torch_cuda"], 0)


@contextlib.contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def equal_state(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (
            torch.is_tensor(a)
            and torch.is_tensor(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a.detach().cpu(), b.detach().cpu())
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


class CheckpointStore:
    """Frozen base/bases stored once per run; all trainable tensors saved per step.

    Frozen state is checked at each save. A changing buffer must be explicitly
    declared mutable at reference creation. Every checkpoint is independently
    hash-verified before loading; construction never deletes or repairs artifacts.
    """

    def __init__(self, reference_directory):
        self.directory = Path(reference_directory).resolve()
        meta = json.loads((self.directory / "reference.json").read_text())
        self.reference_sha256 = meta["sha256"]
        path = self.directory / "reference.pt"
        if sha256(path) != self.reference_sha256:
            raise ValueError("Reference checkpoint checksum mismatch")
        self.reference = torch.load(path, map_location="cpu", weights_only=True)

    @classmethod
    def create(cls, directory, model, provenance, extras=None, mutable_buffers=()):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        mutable_buffers = tuple(mutable_buffers)
        known_buffers = dict(model.named_buffers())
        if set(mutable_buffers) - known_buffers.keys():
            raise ValueError("Unknown mutable buffer")
        trainable = sorted(n for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad)
        if not trainable:
            raise ValueError("Training checkpoint requires trainable parameters")
        state = cpu_copy(model.state_dict())
        changing = sorted(set(trainable) | set(mutable_buffers))
        if set(changing) - state.keys():
            raise ValueError("Changing parameter/buffer missing from model state")
        reference = dict(
            schema_version=1,
            model=state,
            trainable_names=trainable,
            changing_names=changing,
            provenance=provenance,
            extras=cpu_copy(extras or {}),
        )
        digest = save_torch_new(directory / "reference.pt", reference)
        write_json_new(directory / "reference.json", dict(schema_version=1, sha256=digest))
        return cls(directory)

    def validate_model(self, model):
        trainable = sorted(n for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad)
        if trainable != self.reference["trainable_names"]:
            raise ValueError("Trainable parameter set differs from reference")
        state = model.state_dict()
        if state.keys() != self.reference["model"].keys():
            raise ValueError("Model state schema differs from reference")
        changing = set(self.reference["changing_names"])
        for key, value in state.items():
            expected = self.reference["model"][key]
            if torch.is_tensor(value):
                if not torch.is_tensor(expected) or value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(f"Tensor specification changed: {key}")
            if key not in changing and not equal_state(value, expected):
                raise ValueError(f"Frozen reference state changed: {key}")
        return state

    def save(self, directory, model, optimizer, scheduler, stream_state, progress):
        state = self.validate_model(model)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        restart = dict(
            schema_version=1,
            reference_sha256=self.reference_sha256,
            model=cpu_copy({k: state[k] for k in self.reference["changing_names"]}),
            optimizer=cpu_copy(optimizer.state_dict()),
            scheduler=cpu_copy(scheduler.state_dict()),
            stream=cpu_copy(stream_state),
            progress=cpu_copy(progress),
            rng=capture_rng(),
        )
        digest = save_torch_new(directory / "state.pt", restart)
        write_json_new(
            directory / "checkpoint.json",
            dict(schema_version=1, sha256=digest, reference_sha256=self.reference_sha256, step=progress["step"]),
        )
        return directory

    def read(self, directory):
        directory = Path(directory)
        meta = json.loads((directory / "checkpoint.json").read_text())
        if meta["reference_sha256"] != self.reference_sha256:
            raise ValueError("Checkpoint belongs to a different immutable reference")
        if sha256(directory / "state.pt") != meta["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
        if state["reference_sha256"] != self.reference_sha256 or state["progress"]["step"] != meta["step"]:
            raise ValueError("Checkpoint provenance/step mismatch")
        if set(state["model"]) != set(self.reference["changing_names"]):
            raise ValueError("Checkpoint changed-parameter schema mismatch")
        return state

    def restore(self, directory, model, optimizer=None, scheduler=None, stream=None, restore_random_state=True):
        # Validate architecture/trainable set before touching user-visible state.
        if (optimizer is None) != (scheduler is None):
            raise ValueError("Optimizer and scheduler must be restored together")
        trainable = sorted(n for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad)
        if trainable != self.reference["trainable_names"]:
            raise ValueError("Restore model has a different trainable parameter set")
        state = self.read(directory)
        target = model.state_dict()
        if target.keys() != self.reference["model"].keys():
            raise ValueError("Restore model has a different state schema")
        for key, value in target.items():
            saved = self.reference["model"][key]
            if torch.is_tensor(value) and (
                not torch.is_tensor(saved) or value.shape != saved.shape or value.dtype != saved.dtype
            ):
                raise ValueError(f"Restore tensor specification mismatch: {key}")
        model.load_state_dict({**self.reference["model"], **state["model"]}, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        if stream is not None:
            stream.load_state_dict(state["stream"])
        if restore_random_state:
            restore_rng(state["rng"])
        return cpu_copy(state["progress"])
