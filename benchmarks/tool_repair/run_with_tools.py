"""Repeated Codex repair trials with local tools and independent JAX checks."""

import argparse
import concurrent.futures
import inspect
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import time

import jax
import jaxlib

from cases import CASES

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/repair"))
from run import ARMS, MODEL, run_python, safe_candidate
from jax_diagnostics import diagnose

PROMPT = """Repair solution.py for the JAX 0.10.0 behavior contract below.
You may inspect local files, edit solution.py, and use shell tools in this workspace.
Use ./check to run the provided public smoke test. Leave check and smoke.py unchanged.
Please run ./check after repairing the module. Other acceptance checks will run separately.
Keep dependencies, validation and JIT enabled. Use only the supplied material and local
tools. Do not browse, search other directories, download resources, or create other agents.
When finished, leave the patch in solution.py and summarize the work briefly.
"""


def context(case, traceback):
    obj = jax
    for segment in case["docs"].split(".")[1:]:
        obj = getattr(obj, segment)
    docs = (inspect.getdoc(obj) or "")[:2400]
    assert docs
    skill = (ROOT / "SKILL.md").read_text()
    route = diagnose(traceback)
    if route:
        guide, anchor = route.link.split("#")
        header = f"## {anchor.upper()}"
        section = (ROOT / guide).read_text().split(header, 1)[1].split("\n## ", 1)[0]
        skill += f"\n\nRouted guide\n{header}{section}"
    return {"raw": "", "docs": f"\n\nUpstream JAX documentation\n{docs}",
            "skill": f"\n\nJAX Diagnostics skill\n{skill}"}, route.id if route else None


def preflight():
    prompts = {}
    with tempfile.TemporaryDirectory(prefix="jax-tool-preflight-") as temp:
        directory = Path(temp)
        for name, case in CASES.items():
            (directory / "solution.py").write_text(case["program"])
            (directory / "smoke.py").write_text(case["smoke"])
            broken = run_python(Path(sys.executable), directory, "smoke.py")
            assert broken.returncode and case["error"] in broken.stderr, (name, broken.stderr)
            traceback = broken.stderr.replace(str(directory.resolve()) + "/", "")
            traceback = traceback.replace(str(Path(sys.prefix)), "<venv>")
            arms, route = context(case, traceback)

            (directory / "solution.py").write_text(case["reference"])
            (directory / "verify.py").write_text(case["verify"])
            for test in ("smoke.py", "verify.py"):
                fixed = run_python(Path(sys.executable), directory, test)
                assert fixed.returncode == 0, (name, fixed.stderr)
            prompts[name] = {arm: f"{PROMPT}\n{case['task']}\n\nOriginal failing check\n{traceback}{extra}"
                             for arm, extra in arms.items()}
            print(f"Preflight passed: {name}; skill route {route}", flush=True)
    return prompts


def trial(case, arm, repeat, prompt, output):
    label = f"{case}-{arm}-{repeat}"
    row = {"case": case, "arm": arm, "repeat": repeat, "verified": False}
    with tempfile.TemporaryDirectory(prefix="jax-tool-trial-") as temp:
        directory = Path(temp)
        (directory / "solution.py").write_text(CASES[case]["program"])
        smoke = directory / "smoke.py"
        smoke.write_text(CASES[case]["smoke"])
        check = directory / "check"
        check_script = (
            '#!/bin/sh\n'
            'printf "%s\\n" run >> .check-runs\n'
            'export JAX_PLATFORMS=cpu\n'
            f'exec {shlex.quote(sys.executable)} -B smoke.py\n'
        )
        check.write_text(check_script)
        check.chmod(0o755)
        count = directory / ".check-runs"
        command = [
            "codex", "exec", "--cd", str(directory), "--skip-git-repo-check",
            "--ignore-user-config", "--ignore-rules", "--ephemeral", "--sandbox", "workspace-write",
            "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="low"',
            "-m", MODEL, "--json", "-",
        ]
        started = time.monotonic()
        try:
            executed = subprocess.run(command, input=prompt, text=True, capture_output=True, timeout=150)
        except subprocess.TimeoutExpired:
            return {**row, "status": "timeout", "seconds": round(time.monotonic() - started, 3),
                    "test_runs": len(count.read_text().splitlines()) if count.exists() else 0}
        row["seconds"] = round(time.monotonic() - started, 3)
        row["test_runs"] = len(count.read_text().splitlines()) if count.exists() else 0
        events = [json.loads(line) for line in executed.stdout.splitlines() if line.startswith("{")]
        (output / f"{label}.jsonl").write_text(executed.stdout)
        commands = [event["item"] for event in events if event.get("type") == "item.completed"
                    and event["item"]["type"] == "command_execution"]
        row["shell_commands"] = len(commands)
        row["usage"] = next((event["usage"] for event in events if event.get("type") == "turn.completed"), None)
        if executed.returncode or row["usage"] is None:
            return {**row, "status": "agent_error"}
        if smoke.read_text() != CASES[case]["smoke"] or check.read_text() != check_script:
            return {**row, "status": "modified_test"}
        source = (directory / "solution.py").read_text()
        (output / f"{label}-solution.py").write_text(source)
        try:
            if not safe_candidate(source):
                return {**row, "status": "unexpected_import"}
        except SyntaxError:
            return {**row, "status": "invalid_python"}
        # The agent's workspace is discarded before the verifier runs.
    with tempfile.TemporaryDirectory(prefix="jax-tool-verify-") as temp:
        directory = Path(temp)
        (directory / "solution.py").write_text(source)
        (directory / "verify.py").write_text(CASES[case]["verify"])
        try:
            checked = run_python(Path(sys.executable), directory, "verify.py")
        except subprocess.TimeoutExpired:
            return {**row, "status": "verification_timeout"}
        (output / f"{label}-verification.txt").write_text(checked.stdout + checked.stderr)
        return {**row, "verified": checked.returncode == 0,
                "status": "passed" if checked.returncode == 0 else "failed_checks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    assert jax.__version__ == jaxlib.__version__ == "0.10.0" and jax.default_backend() == "cpu"
    prompts = preflight()
    if args.preflight_only:
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    schedule = [(case, arm, repeat) for repeat in (1, 2) for case in CASES for arm in ARMS]
    random.Random(20261122).shuffle(schedule)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(trial, case, arm, repeat, prompts[case][arm], output): position
                   for position, (case, arm, repeat) in enumerate(schedule)}
        for done in concurrent.futures.as_completed(pending):
            row = {**done.result(), "schedule_position": pending[done]}
            results.append(row)
            print(json.dumps(row), flush=True)
            (output / "results.json").write_text(json.dumps(
                sorted(results, key=lambda item: item["schedule_position"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
