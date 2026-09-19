"""Guards on the CLI entry points themselves.

A shadowed definition is invisible to ordinary unit tests: the module imports, every function under test
passes, and the command line silently runs a different function than the one the tests exercise. That is
what happened to ``pilot.train``, where a second definition later in the file shadowed the admission-aware
one, so protocol-driven arguments were never materialized and admission never ran. These tests assert the
properties of the *bound* callables and of the file as a whole.
"""

import argparse
import ast
import inspect
from pathlib import Path

import pytest

from notebooks.iclr.decoder_pilot import choice_runner, interaction_runner, pilot, subspace_runner

MODULES = (pilot, subspace_runner, interaction_runner, choice_runner)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_no_top_level_definition_is_shadowed(module):
    """Two defs with one name means the command line runs something the tests never see."""
    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"{module.__name__} defines these more than once, later wins: {duplicates}"


def test_the_bound_pilot_train_is_the_admission_aware_one():
    source = inspect.getsource(pilot.train)
    assert "resolve_configuration(args)" in source, "the live train() must resolve its configuration"
    assert "plan.validate_admission" in source, "the live train() must run admission before training"
    assert "run_steps(" in source, "the live train() must actually reach the engine"


@pytest.mark.parametrize("module, admission", [
    (subspace_runner, "subspace_plan.validate_admission"),
    (interaction_runner, "ip.validate_admission"),
    (choice_runner, "cp.validate_admission"),
], ids=lambda x: getattr(x, "__name__", str(x)).rsplit(".", 1)[-1])
def test_every_bound_train_runs_admission_and_reaches_the_engine(module, admission):
    source = inspect.getsource(module.train)
    assert admission in source, f"{module.__name__}.train must run admission before training"
    assert "run_steps(" in source, f"{module.__name__}.train must reach the engine"


def _bare_namespace(stage):
    return argparse.Namespace(protocol=None, stage=stage, entry_id=None, acknowledge_unregistered=False,
                              alpha="unset", **{name: None for name in pilot.PROTOCOL_CONTROLLED})


@pytest.mark.parametrize("stage", ["tuning", "calibration", "confirmation"])
def test_a_registered_stage_cannot_run_without_a_protocol(stage):
    with pytest.raises(ValueError, match="require --protocol"):
        pilot.resolve_configuration(_bare_namespace(stage))


def test_an_unregistered_pilot_must_acknowledge_itself():
    with pytest.raises(ValueError, match="acknowledge-unregistered"):
        pilot.resolve_configuration(_bare_namespace("pilot"))


def test_every_cli_subcommand_maps_to_a_real_callable():
    """The dispatch table must not name a function that no longer exists or was renamed."""
    for module in MODULES:
        tree = ast.parse(Path(inspect.getfile(module)).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and node.keys and all(isinstance(k, ast.Constant) for k in node.keys):
                names = [k.value for k in node.keys]
                values = node.values
                if all(isinstance(v, ast.Name) for v in values) and set(names) <= {"prepare", "feasibility", "train", "reference", "audit"}:
                    for name, value in zip(names, values):
                        assert hasattr(module, value.id), f"{module.__name__} dispatches {name!r} to missing {value.id!r}"


@pytest.mark.parametrize("module, admission", [
    (subspace_runner, "subspace_plan.validate_admission"),
    (choice_runner, "cp.validate_admission"),
], ids=lambda x: getattr(x, "__name__", str(x)).rsplit(".", 1)[-1])
def test_every_bound_reference_runs_admission_and_trains_nothing(module, admission):
    """The frozen anchor is a registered run too: it is admitted, and it must not train."""
    source = inspect.getsource(module.reference)
    assert admission in source, f"{module.__name__}.reference must run admission"
    assert "run_steps(" not in source, f"{module.__name__}.reference must not train"
    assert "requires_grad_(False)" in source, f"{module.__name__}.reference must freeze the model"


def test_every_plan_module_that_completes_runs_checks_its_validation_keys():
    """`complete` is the last gate before a run counts as evidence, in every study."""
    from notebooks.iclr.decoder_pilot import choice_plan, interaction_plan, subspace_plan

    for module in (subspace_plan, interaction_plan, choice_plan):
        source = Path(inspect.getfile(module)).read_text()
        assert "did not pass whole-run validation" in source or "VALIDATION_KEYS" in source, \
            f"{module.__name__} completes runs without checking whole-run validation"
