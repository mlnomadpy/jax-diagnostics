"""Run a paired, tool-free Codex repair study and save per-trial measurements."""

import argparse
import ast
from collections import defaultdict
import concurrent.futures
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import jax
import jaxlib

from cases import CASES

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-6.1-sol"
ARMS = ("raw", "docs", "skill")
SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}
PREFIX = (
    "Repair the Python module below for JAX 0.10.0. Return the complete module in the code field. "
    "Use only the context in this message. Do not use tools, browse, or inspect files. "
    "Preserve the requested function interface and semantics, and do not disable JAX checks "
    "or compilation. Your code will be tested on additional inputs."
)


def run_python(python, directory, filename):
    # Candidate programs have no network or file-write access outside their scratch area.
    profile = (
        '(version 1)(allow default)(deny network*)(deny file-write*)'
        f'(allow file-write* (subpath "{directory.resolve()}"))'
        "(deny file-read* "
        + " ".join(f'(subpath {json.dumps(str(Path.home() / part))})'
                   for part in (".ssh", ".codex", ".config"))
        + ")"
    )
    return subprocess.run(
        ["sandbox-exec", "-p", profile, str(python), "-B", filename],
        cwd=directory, text=True, capture_output=True, timeout=25,
        env={"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu", "HOME": str(directory),
             "TMPDIR": str(directory), "PYTHONHASHSEED": "0"},
    )


def build_context(case, traceback):
    from jax_diagnostics import diagnose

    obj = jax
    for part in case["docs"].split(".")[1:]:
        obj = getattr(obj, part)
    docs = (inspect.getdoc(obj) or "")[:2400]
    assert docs
    skill = (ROOT / "SKILL.md").read_text()
    route = diagnose(traceback)
    if route is not None:
        guide, anchor = route.link.split("#", 1)
        content = (ROOT / guide).read_text()
        heading = f"## {anchor.upper()}"
        section = content.split(heading, 1)[1].split("\n## ", 1)[0]
        skill += f"\n\nRouted reference\n{heading}{section}"
    return {"raw": "", "docs": f"\n\nUpstream JAX documentation\n{docs}",
            "skill": f"\n\nJAX Diagnostics skill\n{skill}"}


def safe_candidate(source):
    # Candidate verification just needs JAX and NumPy. Keep accidental unrelated
    # Python imports out of this small benchmark.
    modules = {"jax", "numpy", "functools", "math", "typing", "operator"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in modules for alias in node.names):
                return False
        if isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in modules:
                return False
    return True


def trial(task, arm, prompts, python, schema, data):
    with tempfile.TemporaryDirectory(prefix="jax-repair-trial-") as temp:
        directory = Path(temp)
        started = time.monotonic()
        command = [
            "codex", "exec", "--cd", str(directory), "--skip-git-repo-check",
            "--ignore-user-config", "--ignore-rules", "--ephemeral", "--sandbox", "read-only",
            "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="low"',
            "-m", MODEL, "--json", "--output-schema", str(schema), "-",
        ]
        row = {"case": task, "arm": arm, "verified": False}
        try:
            completed = subprocess.run(
                command, input=prompts[task][arm], text=True, capture_output=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return {**row, "status": "timeout", "seconds": round(time.monotonic() - started, 3)}
        seconds = round(time.monotonic() - started, 3)
        row["seconds"] = seconds
        events = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
        (data / f"{task}-{arm}.jsonl").write_text(completed.stdout)
        messages = [e["item"] for e in events if e.get("type") == "item.completed"]
        usage = next((e["usage"] for e in events if e.get("type") == "turn.completed"), None)
        if completed.returncode or usage is None:
            return {**row, "status": "agent_error"}
        row["usage"] = usage
        if any(item["type"] not in {"agent_message", "reasoning"} for item in messages):
            return {**row, "status": "tool_use_or_harness_warning"}
        try:
            answer = next(item["text"] for item in reversed(messages) if item["type"] == "agent_message")
            source = json.loads(answer)["code"]
            if not safe_candidate(source):
                return {**row, "status": "unexpected_import"}
        except (StopIteration, ValueError, SyntaxError, KeyError, TypeError):
            return {**row, "status": "invalid_output"}
        (data / f"{task}-{arm}-solution.py").write_text(source)
        return verify_proposal(row, source, python, directory)


def verify_proposal(row, source, python, directory):
    (directory / "solution.py").write_text(source)
    (directory / "verify.py").write_text(CASES[row["case"]]["verify"])
    try:
        verification = run_python(python, directory, "verify.py")
    except subprocess.TimeoutExpired:
        return {**row, "status": "verification_timeout", "verified": False}
    result = {**row, "verified": verification.returncode == 0,
              "status": "passed" if verification.returncode == 0 else "failed_checks"}
    if verification.returncode:
        lines = verification.stderr.strip().splitlines()
        result["failure"] = lines[-1] if lines else f"Exit status {verification.returncode}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    assert jax.__version__ == jaxlib.__version__ == "0.10.0"
    assert jax.default_backend() == "cpu"
    python = Path(sys.executable)
    data = args.output.resolve()
    data.mkdir(parents=True, exist_ok=False)
    schema = data / "answer-schema.json"
    schema.write_text(json.dumps(SCHEMA))
    prompts = defaultdict(dict)
    for name, case in CASES.items():
        with tempfile.TemporaryDirectory(prefix="jax-repair-repro-") as temp:
            directory = Path(temp)
            (directory / "solution.py").write_text(case["program"])
            (directory / "repro.py").write_text(case["repro"])
            repro = run_python(python, directory, "repro.py")
            assert repro.returncode and case["expected_error"] in repro.stderr, (name, repro.stderr)
            traceback = re.sub(re.escape(str(directory.resolve())) + "/", "", repro.stderr)
            traceback = traceback.replace(str(Path(sys.prefix).resolve()), "<venv>")
            first = traceback.index("Traceback")
            traceback = traceback[first:]
            for arm, extra in build_context(case, traceback).items():
                prompt = (
                    f"{PREFIX}\n\nRequirements\n{case['task']}\n\n"
                    f"Module\n{case['program']}\nReproduction command: python repro.py\n"
                    f"{case['repro']}\nTraceback\n{traceback}{extra}"
                )
                prompts[name][arm] = prompt
                (data / f"{name}-{arm}-prompt.txt").write_text(prompt)
            print(f"Reproduced {name}", flush=True)
    if args.check_only:
        return
    schedule = [(case, arm) for case in CASES for arm in ARMS]
    random.Random(20261121).shuffle(schedule)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {
            pool.submit(trial, case, arm, prompts, python, schema, data): position
            for position, (case, arm) in enumerate(schedule)
        }
        results = []
        for job in concurrent.futures.as_completed(jobs):
            row = {**job.result(), "schedule_position": jobs[job]}
            results.append(row)
            print(json.dumps(row), flush=True)
            (data / "results.json").write_text(json.dumps(sorted(results, key=lambda r: r["schedule_position"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
