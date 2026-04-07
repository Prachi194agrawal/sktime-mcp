"""
tests/bug_demo.py
=====================================================================
Bug #2 Demo: load_data_source_async passes estimator_handle=""
and no estimator_name to create_job(), polluting list_jobs() output.

Affected file:
  src/sktime_mcp/runtime/executor.py  (load_data_source_async)

Run with:
    python tests/bug_demos/test_bug2_missing_estimator_name.py
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    job_id:           str
    job_type:         str
    estimator_handle: Optional[str]
    status:           JobStatus     = JobStatus.PENDING
    created_at:       datetime      = field(default_factory=datetime.now)
    estimator_name:   Optional[str] = None
    dataset_name:     Optional[str] = None
    total_steps:      int           = 0
    completed_steps:  int           = 0
    result:           Any           = None
    errors:           list          = field(default_factory=list)


class FakeJobManager:
    def __init__(self):
        self._jobs: dict[str, JobInfo] = {}

    def create_job(self, job_type, estimator_handle,
                   dataset_name=None, total_steps=3,
                   estimator_name=None, **kwargs) -> str:
        import uuid
        jid = str(uuid.uuid4())[:8]
        self._jobs[jid] = JobInfo(
            job_id=jid,
            job_type=job_type,
            estimator_handle=estimator_handle,
            estimator_name=estimator_name,
            dataset_name=dataset_name,
            total_steps=total_steps,
        )
        return jid

    def list_jobs(self):
        return list(self._jobs.values())


# ══════════════════════════════════════════════════════════════════════
# BUGGY — mirrors exact code in executor.py load_data_source_async
# ══════════════════════════════════════════════════════════════════════
def buggy_create_job(jm, source_type):
    return jm.create_job(
        job_type="data_loading",
        estimator_handle="",       # BUG: empty string
        dataset_name=source_type,
        total_steps=3,
        # estimator_name intentionally omitted
    )


# ══════════════════════════════════════════════════════════════════════
# FIXED
# ══════════════════════════════════════════════════════════════════════
def fixed_create_job(jm, source_type):
    return jm.create_job(
        job_type="data_loading",
        estimator_handle=None,
        estimator_name=f"data_source:{source_type}",
        dataset_name=source_type,
        total_steps=3,
    )


def test_buggy():
    print("\n" + "="*60)
    print("TEST 1: BUGGY pattern")
    print("="*60)
    jm = FakeJobManager()
    buggy_create_job(jm, "file")
    job = jm.list_jobs()[0]
    print(f"  estimator_handle = {repr(job.estimator_handle)}")
    print(f"  estimator_name   = {repr(job.estimator_name)}")
    handle_is_empty = job.estimator_handle == ""
    name_is_none    = job.estimator_name is None
    if handle_is_empty:
        print('  [RESULT] BUG CONFIRMED: estimator_handle = ""')
    if name_is_none:
        print("  [RESULT] BUG CONFIRMED: estimator_name = None")
    return handle_is_empty and name_is_none


def test_fixed():
    print("\n" + "="*60)
    print("TEST 2: FIXED pattern")
    print("="*60)
    jm = FakeJobManager()
    fixed_create_job(jm, "file")
    job = jm.list_jobs()[0]
    print(f"  estimator_handle = {repr(job.estimator_handle)}")
    print(f"  estimator_name   = {repr(job.estimator_name)}")
    handle_is_none      = job.estimator_handle is None
    name_is_descriptive = job.estimator_name == "data_source:file"
    if handle_is_none:
        print("  [RESULT] FIX CONFIRMED: estimator_handle = None")
    if name_is_descriptive:
        print(f"  [RESULT] FIX CONFIRMED: estimator_name = {repr(job.estimator_name)}")
    return handle_is_none and name_is_descriptive


def test_list_jobs_comparison():
    print("\n" + "="*60)
    print("TEST 3: list_jobs() side-by-side comparison")
    print("="*60)
    jm = FakeJobManager()
    jm.create_job(                          # normal fit_predict job
        job_type="fit_predict",
        estimator_handle="est_abc123",
        estimator_name="NaiveForecaster",
        dataset_name="airline",
    )
    buggy_create_job(jm, "sql")             # buggy data job
    fixed_create_job(jm, "file")            # fixed data job

    print(f"\n  {'JOB TYPE':<20} {'HANDLE':<20} {'NAME'}")
    print(f"  {'-'*20} {'-'*20} {'-'*30}")
    for j in jm.list_jobs():
        print(f"  {j.job_type:<20} {repr(j.estimator_handle):<20} {repr(j.estimator_name)}")
    print()
    print("  BUGGY row: '' handle and None name — confusing for LLM clients")
    print("  FIXED row: None handle, 'data_source:file' — clear and parseable")


def test_real_executor():
    print("\n" + "="*60)
    print("TEST 4: Real executor (requires sktime_mcp installed)")
    print("="*60)
    try:
        from pathlib import Path
        from sktime_mcp.runtime.async_runner import submit_coroutine
        from sktime_mcp.runtime.executor import Executor
        from sktime_mcp.runtime.jobs import get_job_manager

        csv_path = Path("/tmp/bug2_demo.csv")
        csv_path.write_text(
            "date,value\n2024-01-01,10\n2024-01-02,12\n"
            "2024-01-03,11\n2024-01-04,15\n2024-01-05,13\n"
        )
        executor = Executor()
        jm = get_job_manager()
        future = submit_coroutine(executor.load_data_source_async({
            "type": "file", "path": str(csv_path),
            "time_column": "date", "target_column": "value",
        }))
        future.result(timeout=15)
        load_jobs = [j for j in jm.list_jobs() if j.job_type == "data_loading"]
        latest = max(load_jobs, key=lambda j: j.created_at)
        print(f"  estimator_handle = {repr(latest.estimator_handle)}")
        print(f"  estimator_name   = {repr(latest.estimator_name)}")
        if latest.estimator_handle == "":
            print("  [RESULT] BUG PRESENT in real executor")
        elif latest.estimator_handle is None and latest.estimator_name:
            print("  [RESULT] FIXED in real executor!")
    except ImportError:
        print("  sktime_mcp not importable — skipping real executor check.")


if __name__ == "__main__":
    print("sktime-mcp Bug #2: missing estimator_name in create_job")
    print("Python", sys.version)
    bug_ok = test_buggy()
    fix_ok = test_fixed()
    test_list_jobs_comparison()
    test_real_executor()
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Bug reproduced : {bug_ok}")
    print(f"  Fix works      : {fix_ok}")
    sys.exit(0 if (bug_ok and fix_ok) else 1)