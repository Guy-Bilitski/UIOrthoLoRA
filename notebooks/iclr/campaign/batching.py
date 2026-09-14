"""Deterministic, fingerprinted tensor examples and exactly resumable batch order."""

import hashlib
import json

import torch


class TensorExamples:
    def __init__(self, tensors, sample_ids):
        if not tensors or "labels" not in tensors:
            raise ValueError("Task examples require tensor fields and labels")
        self.tensors = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
        self.sample_ids = tuple(sample_ids)
        self.size = len(self.sample_ids)
        if not self.size or len(set(self.sample_ids)) != self.size:
            raise ValueError("Require nonempty unique sample IDs")
        if any(v.ndim == 0 or v.shape[0] != self.size for v in self.tensors.values()):
            raise ValueError("All fields must have the same example dimension")
        digest = hashlib.sha256(json.dumps(self.sample_ids, ensure_ascii=False).encode())
        for key, tensor in sorted(self.tensors.items()):
            digest.update(json.dumps([key, list(tensor.shape), str(tensor.dtype)]).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        self.fingerprint = digest.hexdigest()

    def batch(self, indices, device):
        return {k: v[indices].to(device) for k, v in self.tensors.items()}


class BatchStream:
    def __init__(self, examples, batch_size, seed):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Positive batch size required")
        self.examples = examples
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch, self.cursor = 0, 0
        self.order = torch.randperm(examples.size, generator=self.generator)

    def next_indices(self):
        if self.cursor == self.examples.size:
            self.epoch += 1
            self.cursor = 0
            self.order = torch.randperm(self.examples.size, generator=self.generator)
        end = min(self.cursor + self.batch_size, self.examples.size)
        result = self.order[self.cursor : end].clone()
        self.cursor = end
        return result

    def state_dict(self):
        return dict(
            fingerprint=self.examples.fingerprint,
            batch_size=self.batch_size,
            epoch=self.epoch,
            cursor=self.cursor,
            order=self.order.clone(),
            generator=self.generator.get_state(),
        )

    def load_state_dict(self, state):
        if state["fingerprint"] != self.examples.fingerprint or state["batch_size"] != self.batch_size:
            raise ValueError("Dataset fingerprint or batch size changed on resume")
        n = self.examples.size
        if not torch.equal(state["order"].sort().values, torch.arange(n)):
            raise ValueError("Invalid saved batch permutation")
        if not 0 <= state["cursor"] <= n or state["epoch"] < 0:
            raise ValueError("Invalid saved batch cursor/epoch")
        self.order = state["order"].clone()
        self.cursor, self.epoch = state["cursor"], state["epoch"]
        self.generator.set_state(state["generator"])
