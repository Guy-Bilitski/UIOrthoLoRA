"""Supervise only a child launched here, with persistent logs and budget leases.

Run the outer controller in the server's scheduler or a dedicated tmux session.
This module never scans process lists, attaches to arbitrary PIDs, or releases a
lease merely because a heartbeat is old. No training is started on import.
"""

from dataclasses import asdict, dataclass
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import time

from .allocation import AllocationLedger
from .artifacts import sha256, utc_now, write_json_new
from .protocol import owned_path


@dataclass(frozen=True)
class MonitorSettings:
    poll_seconds: float
    stale_seconds: float
    checkpoint_grace_seconds: float
    terminate_grace_seconds: float
    maximum_temperature_c: float
    maximum_prelaunch_memory_mib: float

    def validate(self):
        if any(type(x) not in (float, int) or not isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError("Explicit finite positive monitoring thresholds required")
        if self.poll_seconds > 60 or self.stale_seconds < self.poll_seconds:
            raise ValueError("Poll at most every 60 seconds; stale threshold must span a poll")


def gpu_telemetry(gpu_id):
    fields = ["uuid", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw"]
    output = (
        subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(output) != 1:
        raise ValueError("Expected telemetry for exactly one assigned GPU")
    values = [x.strip() for x in output[0].split(",")]
    if len(values) != len(fields):
        raise ValueError("Unexpected GPU telemetry schema")
    result = dict(zip(fields, [values[0]] + [float(x) for x in values[1:]]))
    if any(not isfinite(x) or x < 0 for x in list(result.values())[1:]):
        raise ValueError("Nonfinite or negative GPU telemetry")
    return result


def _last_json_line(path):
    if not path.exists():
        return None
    # Read a bounded tail of only this child's log. Partial lines are expected
    # during writes, and never interpreted as a completed step/terminal state.
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - 256 * 1024))
        payload = file.read()
    lines = payload.split(b"\n")
    for line in reversed(lines[:-1]):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def _write_line(file, value):
    file.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def supervise_owned_worker(
    allocation_directory,
    job_path,
    checkout,
    python_executable,
    *,
    gpu_id,
    gpu_uuid,
    maximum_seconds,
    reserved_bytes,
    monitor_settings,
    synthetic_cpu_test=False,
):
    """A returned receipt proves child exit, not checkpoint/run validity.

    Production command is fixed to campaign.worker. The only alternate command
    is a fixed synthetic CPU fixture module, with all GPUs hidden. The future
    run controller must validate the immutable job and scientific phase gates.
    """
    monitor_settings.validate()
    work_seconds = (
        maximum_seconds
        - monitor_settings.checkpoint_grace_seconds
        - monitor_settings.terminate_grace_seconds
        - 2 * monitor_settings.poll_seconds
    )
    if work_seconds <= 0:
        raise ValueError("Total reservation must include positive work time plus checkpoint/termination grace")
    book = AllocationLedger(allocation_directory)
    book.resources.validate_training()
    job_path = owned_path(book.root, job_path)
    run_directory = job_path.parent
    checkout = Path(checkout).resolve()
    module = (
        "notebooks.iclr.campaign.tests.supervisor_fixture" if synthetic_cpu_test else "notebooks.iclr.campaign.worker"
    )
    if not (checkout / (module.replace(".", "/") + ".py")).is_file():
        raise ValueError("Required campaign worker entry point has not been implemented")
    job = json.loads(job_path.read_text())
    if Path(job["run_directory"]).resolve() != run_directory or not job.get("run_id"):
        raise ValueError("Job must name its own immutable run directory and ID")
    if job.get("synthetic_cpu_test", False) is not synthetic_cpu_test:
        raise ValueError("Synthetic and production job namespaces cannot be mixed")
    if not synthetic_cpu_test:
        if gpu_id not in book.resources.assigned_gpu_ids:
            raise ValueError("Physical GPU is not assigned")
        if Path(python_executable).absolute() != checkout / ".venv/bin/python":
            raise ValueError("Use the campaign's pinned Python environment")
        baseline = gpu_telemetry(gpu_id)
        if baseline["uuid"] != gpu_uuid:
            raise ValueError("Physical GPU UUID changed")
        if baseline["memory.used"] > monitor_settings.maximum_prelaunch_memory_mib:
            raise ValueError("Assigned GPU is already occupied; do not inspect or interfere with that job")
    else:
        baseline = None
    lease = book.reserve(
        run_id=job["run_id"],
        run_directory=run_directory,
        gpu_id=gpu_id,
        maximum_seconds=maximum_seconds,
        reserved_bytes=reserved_bytes,
    )
    command = [str(python_executable), "-u", "-m", module, "--job", str(job_path)]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "" if synthetic_cpu_test else gpu_uuid,
        "PYTHONPATH": str(checkout) + os.pathsep + str(checkout / "src"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "ICLR_ALLOCATION_DIRECTORY": str(book.directory),
        "ICLR_LEASE_ID": lease["lease_id"],
        "ICLR_GPU_ID": str(gpu_id),
        "ICLR_GPU_UUID": "" if synthetic_cpu_test else gpu_uuid,
        "ICLR_MAXIMUM_SECONDS": str(maximum_seconds),
        "ICLR_WORK_SECONDS": str(work_seconds),
    }
    launch_record = dict(
        created_utc=utc_now(),
        job_sha256=sha256(job_path),
        command=command,
        checkout=str(checkout),
        physical_gpu=gpu_id,
        gpu_uuid=gpu_uuid,
        visible_devices=env["CUDA_VISIBLE_DEVICES"],
        lease_id=lease["lease_id"],
        monitor_settings=asdict(monitor_settings),
        baseline=baseline,
        synthetic_cpu_test=synthetic_cpu_test,
        work_seconds=work_seconds,
    )
    process = None
    settled = False
    started = time.monotonic()
    requested_at = terminated_at = None
    reasons = []
    try:
        write_json_new(
            run_directory / "supervisor_launch.json",
            launch_record,
        )
        with (
            (run_directory / "worker.log").open("xb") as output,
            (run_directory / "monitor.jsonl").open("x") as monitor,
        ):
            process = subprocess.Popen(
                command,
                cwd=checkout,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json_new(
                run_directory / "owned_child.json",
                dict(
                    pid=process.pid,
                    started_utc=utc_now(),
                    command=command,
                    note="Informational PID only; never attach/kill or settle a lease using this file alone",
                ),
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                wall_now = time.time()
                watched = [run_directory / x for x in ("worker.log", "heartbeat.jsonl", "engine/steps.jsonl")]
                last_write = max((p.stat().st_mtime for p in watched if p.exists()), default=wall_now)
                age = max(0, wall_now - last_write)
                latest = _last_json_line(run_directory / "engine/steps.jsonl")
                alerts = []
                telemetry = None
                if not synthetic_cpu_test:
                    try:
                        telemetry = gpu_telemetry(gpu_id)
                    except (subprocess.SubprocessError, ValueError, OSError) as exc:
                        alerts.append("telemetry_unavailable:" + type(exc).__name__)
                    if telemetry is not None:
                        if telemetry["uuid"] != gpu_uuid:
                            reasons.append("gpu_identity_changed")
                        if telemetry["temperature.gpu"] >= monitor_settings.maximum_temperature_c:
                            reasons.append("temperature_limit")
                        if latest is not None and telemetry["utilization.gpu"] == 0:
                            alerts.append("zero_utilization_sample_may_be_checkpoint_or_diagnostics")
                state = book.snapshot()
                if state["actual_bytes"] > book.resources.storage_allowance_gib * 2**30:
                    reasons.append("allocation_storage_exhausted")
                if state["filesystem_free_bytes"] < book.manifest["disk_safety_margin_gib"] * 2**30:
                    reasons.append("filesystem_safety_margin")
                if latest is not None:
                    for key in ("task_loss", "regularization_loss", "gradient_norm_before_clipping"):
                        value = latest.get(key)
                        if not isinstance(value, (int, float)) or not isfinite(value):
                            reasons.append("nonfinite_or_missing_step_health:" + key)
                if age > monitor_settings.stale_seconds:
                    alerts.append("stalled_progress_log")
                if elapsed >= work_seconds:
                    reasons.append("worker_time_reservation_reached")
                stop_path = run_directory / "stop_request.json"
                if stop_path.exists() and requested_at is None:
                    requested_at = time.monotonic()
                    reasons.append("external_stop_request")
                if reasons and requested_at is None:
                    requested_at = time.monotonic()
                    write_json_new(stop_path, dict(created_utc=utc_now(), reasons=sorted(set(reasons))))
                if (
                    requested_at is not None
                    and time.monotonic() - requested_at >= monitor_settings.checkpoint_grace_seconds
                ):
                    if terminated_at is None:
                        process.terminate()  # Exact Popen child created above, never an arbitrary PID.
                        terminated_at = time.monotonic()
                        alerts.append("sent_sigterm_to_owned_child")
                    elif time.monotonic() - terminated_at >= monitor_settings.terminate_grace_seconds:
                        process.kill()
                        alerts.append("sent_sigkill_to_unresponsive_owned_child")
                _write_line(
                    monitor,
                    dict(
                        observed_utc=utc_now(),
                        elapsed_seconds=elapsed,
                        pid=process.pid,
                        progress_age_seconds=age,
                        latest_step=latest,
                        gpu=telemetry,
                        storage_actual_bytes=state["actual_bytes"],
                        filesystem_free_bytes=state["filesystem_free_bytes"],
                        alerts=alerts,
                        stop_reasons=sorted(set(reasons)),
                    ),
                )
                # The persistent process sleeps briefly; its interactive caller
                # need not block. Production polls are bounded to <=60 seconds.
                time.sleep(monitor_settings.poll_seconds)
            returncode = process.wait()
        elapsed = time.monotonic() - started
        settlement = book.settle(
            lease["lease_id"], lease["ownership_token"], observed_exit_code=returncode, measured_worker_seconds=elapsed
        )
        settled = True
        receipt = dict(
            execution_status="child_exited",
            returncode=returncode,
            pid=process.pid,
            elapsed_seconds=elapsed,
            settlement=settlement,
            stop_reasons=sorted(set(reasons)),
            scientific_completion_asserted=False,
            ended_utc=utc_now(),
        )
        write_json_new(run_directory / "execution_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        # If monitoring itself fails, leave the child safely unmodified. Its
        # original time limit/stop-file checks still apply. Do not release the
        # reservation or pretend that an observation failure means child exit.
        terminal = process is None or process.poll() is not None
        if terminal and not settled:
            book.settle(
                lease["lease_id"],
                lease["ownership_token"],
                observed_exit_code=127 if process is None else process.returncode,
                measured_worker_seconds=time.monotonic() - started,
            )
        write_json_new(
            run_directory / "supervisor_error.json",
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                created_utc=utc_now(),
                child_pid=None if process is None else process.pid,
                child_verified_terminal=terminal,
                lease_left_active=not terminal,
            ),
        )
        raise
