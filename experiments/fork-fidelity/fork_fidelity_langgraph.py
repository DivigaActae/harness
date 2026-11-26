#!/usr/bin/env python3
# Copyright 2026 Diviga
# SPDX-License-Identifier: Apache-2.0

"""
LangGraph fork-fidelity suite — LLM-judged evidence that the "native"
LangGraph fork path (`ActaeCheckpointSaver.fork_thread`) produces output
equivalent to a no-fork re-run.

The methodology is identical to `fork_fidelity_suite.py` (see
`docs/FORK_FIDELITY.md`): the same five scenarios, the same judge rubric, the
same two claims per scenario:

  1. fork fidelity >= threshold             (fork output equivalent to baseline)
  2. fork fidelity >= rerun fidelity - tolerance   (forking not worse than re-running)

The difference is HOW the fork happens. Instead of the AgentSession-based
`run_fork` (which forks the channel and re-runs the tail with a manual loop),
each scenario step becomes ONE LangGraph node, the full pipeline is a linear
StateGraph compiled against an `ActaeCheckpointSaver`, and the fork uses the
framework-native path:

    hist = [t async for t in graph.aget_state_history(cfg)]   # newest-first
    ... find the checkpoint whose state contains exactly `fork_at_step`
    ... step keys (i.e. the checkpoint right after node `fork_at_step`) ...
    fork_cfg = await saver.fork_thread(
        {"configurable": {"thread_id": base, "checkpoint_id": cp_id}},
        new_thread_id=base + "-fork", reason="fidelity fork at step N")
    res = await fork_graph.ainvoke(None, fork_cfg)             # tail re-runs

The inherited prefix (steps 1..N) is verified byte-identical
(`prefix_identical`), and fork provenance is verified on the server side:
the fork channel's first event is `fork.started` and the immutable fork
receipt reports a restorable boundary with equal requested/resolved cursors
and a non-empty source-state SHA-256.

Run:
    # Plumbing only (deterministic fake pipeline + judge, no API key):
    python examples/fork_fidelity_langgraph.py --dry-run

    # Real evidence run (DeepSeek key from actae/.env; ~10-20 min):
    python examples/fork_fidelity_langgraph.py --repeats 3 --judge-reps 1

Reports are written to `examples/fidelity-reports/fork-fidelity-langgraph-<run-id>.json`.
Exit code is 0 (all scenarios passed) or 1 (any failed / hard error).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from openai import AsyncOpenAI

from actae_client import ActaeClient
from actae_client.adapters.langgraph import ActaeCheckpointSaver, _HAS_LANGGRAPH
from actae_client.errors import RateLimitError
from langgraph.graph import END, START, StateGraph

from fork_fidelity_lib import (
    DeterministicJudge,
    FidelityComparison,
    FidelityError,
    Judge,
    LLMJudge,
    ScenarioResult,
    Scenario,
    aggregate,
    aggregate_comparison,
    prefix_identical,
    scenario_passed,
    tail_text,
    _is_transient,
)
from fork_fidelity_scenarios import ALL_SCENARIOS, scenarios_by_name

if not _HAS_LANGGRAPH:
    raise SystemExit("langgraph is not installed (pip install langgraph)")

ACTAE_URL = os.environ.get("ACTAE_URL", "http://localhost:8002")
ACTAE_API_KEY = os.environ.get("ACTAE_API_KEY", "sk-dev-0000000000000000000000")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "fidelity-reports"

DEEPSEEK_API_KEY: Optional[str] = None

ADAPTER = "langgraph"


def _load_deepseek_key() -> None:
    global DEEPSEEK_API_KEY
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        DEEPSEEK_API_KEY = key
        return
    env_path = Path(__file__).resolve().parent.parent / "actae" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                DEEPSEEK_API_KEY = line.split("=", 1)[1].strip().strip("\"'")
                return
    raise SystemExit(
        "DEEPSEEK_API_KEY not found. Set it in actae/.env or the environment "
        "(or run with --dry-run, which needs no API key)."
    )


# ---------------------------------------------------------------------------
# LangGraph pipeline: each scenario step is one linear graph node.
# ---------------------------------------------------------------------------


class Acc(TypedDict):
    """State schema: one `acc` channel holding the accumulated scenario dict."""

    acc: dict


def build_graph(
    saver: ActaeCheckpointSaver,
    sc: Scenario,
    chat: Any,
    model: str,
    dry_run: bool,
    run_tag: str,
    counter: Optional[Dict[str, int]] = None,
):
    """Compile a linear StateGraph with one node per scenario step.

    Every node reads the accumulated `acc` dict, calls the LLM (or emits a
    deterministic dry-run string) for its step key, and returns the FULL
    accumulated dict plus the new key. Because the graph state is the
    accumulated dict (LastValue), each checkpoint stores the whole prefix —
    exactly what `fork_thread` needs to inherit steps 1..N byte-identically.
    """
    g = StateGraph(Acc)

    for i, step in enumerate(sc.steps):
        key = step.key
        node_name = f"n{i + 1}"

        async def node(state, _s=step, _k=key):
            acc = dict(state["acc"])
            user = _s.user(acc)
            if dry_run:
                content = f"[dry-run:{_k}] {_s.system[:80]}"
                if counter is not None:
                    counter["calls"] += 1
                    counter["tokens"] += 60 + len(user) // 4 + 20 + len(content) // 4
            else:
                content = await _node_llm(chat, model, _s, user, counter=counter)
            acc[_k] = content
            return {"acc": acc}

        g.add_node(node_name, node)
        if i == 0:
            g.add_edge(START, node_name)
        else:
            g.add_edge(f"n{i}", node_name)
    g.add_edge(f"n{len(sc.steps)}", END)
    return g.compile(checkpointer=saver)


async def _node_llm(
    chat: Any,
    model: str,
    step: Any,
    user: str,
    *,
    counter: Optional[Dict[str, int]] = None,
    retries: int = 3,
) -> str:
    """One real LLM call for a pipeline node (mirrors fork_fidelity_lib's
    `_llm_text` retry budget; dry-run never reaches here)."""
    for attempt in range(retries):
        try:
            resp = await chat.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": step.system},
                    {"role": "user", "content": user},
                ],
                temperature=step.temperature,
                max_tokens=step.max_tokens,
            )
            usage = getattr(resp, "usage", None)
            if counter is not None:
                counter["calls"] += 1
                counter["tokens"] += (
                    (usage.prompt_tokens if usage else 0)
                    + (usage.completion_tokens if usage else 0)
                )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(min(5 * (2 ** attempt), 30))


def _initial_state(sc: Scenario) -> Dict[str, Any]:
    """The graph input: an `acc` dict with the scenario's live context."""
    acc: Dict[str, Any] = {"topic": sc.topic}
    if sc.input_text:
        acc["_input"] = sc.input_text
    return {"acc": acc}


# ---------------------------------------------------------------------------
# Checkpoint matching: where to fork
# ---------------------------------------------------------------------------


def _find_fork_checkpoint(history: List[Any], sc: Scenario) -> Any:
    """Locate the checkpoint whose state contains exactly `fork_at_step` step
    keys — i.e. the checkpoint recorded right after node `fork_at_step`.

    `aget_state_history` yields checkpoints newest-first and the accumulated
    `acc` dict gains exactly one step key per node, so the key count is
    strictly decreasing along the history. The FIRST tuple with exactly
    `fork_at_step` step keys is therefore the newest checkpoint with that many
    steps — the fork point that inherits steps 1..N and re-runs N+1..M.
    """
    step_keys = [s.key for s in sc.steps]
    for t in history:
        values = t.values or {}
        acc = values.get("acc") or {}
        if sum(1 for k in step_keys if k in acc) == sc.fork_at_step:
            return t
    raise FidelityError(
        f"{sc.name}: no checkpoint in history with exactly "
        f"{sc.fork_at_step} step keys (history depth {len(history)})"
    )


# ---------------------------------------------------------------------------
# Provenance verification
# ---------------------------------------------------------------------------


def _fork_provenance(receipt: Any, fork_channel: str) -> Dict[str, Any]:
    """Validate the server-side fork receipt and return it as a dict.

    Live evidence requires a restorable boundary, equal requested/resolved
    cursors, and a non-empty source-state SHA-256 — a loud FidelityError
    otherwise."""
    if not receipt.restorable:
        raise FidelityError(f"fork {fork_channel}: boundary is not restorable")
    if receipt.requested_cursor != receipt.resolved_cursor:
        raise FidelityError(
            f"fork {fork_channel}: requested cursor {receipt.requested_cursor} "
            f"resolved to {receipt.resolved_cursor}"
        )
    if not receipt.source_state_sha256:
        raise FidelityError(f"fork {fork_channel}: receipt has no source-state fingerprint")
    return {
        "fork_id": receipt.fork_id,
        "source_channel_id": receipt.source_channel_id,
        "restorable": receipt.restorable,
        "requested_cursor": receipt.requested_cursor,
        "resolved_cursor": receipt.resolved_cursor,
        "source_state_version": receipt.source_state_version,
        "source_state_sha256": receipt.source_state_sha256,
        "reproducibility": receipt.reproducibility,
    }


# ---------------------------------------------------------------------------
# One repeat: baseline + re-run + fork via LangGraph
# ---------------------------------------------------------------------------


async def _run_one_repeat(
    actae: ActaeClient,
    sc: Scenario,
    *,
    chat: Any,
    pipeline_model: str,
    dry_run: bool,
    judge: Judge,
    run_id: str,
    repeat_idx: int,
) -> Dict[str, Any]:
    """Run the LangGraph baseline, a full no-fork re-run, and a
    `fork_thread` fork once; judge the tails.

    Channels are `fid-lg-<sc>-<runid>-r<i>` hashed with the thread id
    (ActaeCheckpointSaver's deterministic channel resolution); threads are
    `t-<sc>-<runid>-r<i>`, `...-rerun` and `...-fork`."""
    tag = f"{sc.name}-{run_id}-r{repeat_idx}"
    saver = ActaeCheckpointSaver(actae, channel=f"fid-lg-{tag}")
    base_thread = f"t-{tag}"
    rerun_thread = f"{base_thread}-rerun"
    fork_thread_id = f"{base_thread}-fork"
    init = _initial_state(sc)

    # --- baseline: full graph run on the base thread ---
    base_counter: Dict[str, int] = {"calls": 0, "tokens": 0}
    full = build_graph(saver, sc, chat, pipeline_model, dry_run, tag, base_counter)
    base_state = (await full.ainvoke(init, {"configurable": {"thread_id": base_thread}}))["acc"]
    base_channel = saver.channel_for_config({"configurable": {"thread_id": base_thread}})

    # --- re-run: the no-fork control on a fresh thread ---
    rerun_counter: Dict[str, int] = {"calls": 0, "tokens": 0}
    rerun_graph = build_graph(saver, sc, chat, pipeline_model, dry_run, tag, rerun_counter)
    rerun_state = (await rerun_graph.ainvoke(init, {"configurable": {"thread_id": rerun_thread}}))["acc"]
    rerun_channel = saver.channel_for_config({"configurable": {"thread_id": rerun_thread}})

    # --- fork: checkpoint-matching + framework-native fork_thread ---
    hist = [t async for t in full.aget_state_history({"configurable": {"thread_id": base_thread}})]
    fork_target = _find_fork_checkpoint(hist, sc)
    checkpoint_id = fork_target.config["configurable"]["checkpoint_id"]
    fork_cfg = await saver.fork_thread(
        {"configurable": {"thread_id": base_thread, "checkpoint_id": checkpoint_id}},
        new_thread_id=fork_thread_id,
        reason=f"fidelity fork at step {sc.fork_at_step}",
    )
    fork_channel = saver.channel_for_config(fork_cfg)

    # The fork's first event must be the fork.started lineage marker.
    events = await actae.replay(fork_channel, cursor=0, limit=50)
    if not events or events[0].event_type != "fork.started":
        raise FidelityError(
            f"{sc.name}: fork channel {fork_channel} first event is "
            f"{events[0].event_type if events else 'none'}, expected fork.started"
        )

    fork_counter: Dict[str, int] = {"calls": 0, "tokens": 0}
    fork_graph = build_graph(saver, sc, chat, pipeline_model, dry_run, tag, fork_counter)
    res = await fork_graph.ainvoke(None, fork_cfg)
    fork_state = res["acc"]

    receipt = await actae.get_fork_receipt(fork_channel)
    provenance = _fork_provenance(receipt, fork_channel)
    provenance["forked_checkpoint_id"] = checkpoint_id

    prefix_ok, prefix_mismatches = prefix_identical(base_state, fork_state, sc)
    fork_cmp = await judge.score(tail_text(base_state, sc), tail_text(fork_state, sc))
    rerun_cmp = await judge.score(tail_text(base_state, sc), tail_text(rerun_state, sc))

    calls = base_counter["calls"] + rerun_counter["calls"] + fork_counter["calls"]
    tokens = base_counter["tokens"] + rerun_counter["tokens"] + fork_counter["tokens"]

    return {
        "prefix_ok": prefix_ok,
        "prefix_mismatches": prefix_mismatches,
        "fork": fork_cmp,
        "rerun": rerun_cmp,
        "calls": calls,
        "tokens": tokens,
        "channels": {"baseline": base_channel, "rerun": rerun_channel, "fork": fork_channel},
        "observations": {
            "baseline": {"state": base_state, "tail": tail_text(base_state, sc)},
            "rerun": {"state": rerun_state, "tail": tail_text(rerun_state, sc)},
            "fork": {"state": fork_state, "tail": tail_text(fork_state, sc)},
        },
        "fork_provenance": provenance,
        "langgraph": {
            "thread": fork_thread_id,
            "fork_channel": fork_channel,
            "forked_checkpoint_id": checkpoint_id,
            "fork_at_step": sc.fork_at_step,
            "receipt": provenance,
        },
    }


async def _repeat_with_retry(
    actae: ActaeClient,
    sc: Scenario,
    *,
    chat: Any,
    pipeline_model: str,
    dry_run: bool,
    judge: Judge,
    run_id: str,
    repeat_idx: int,
    retries: int = 2,
) -> Tuple[Dict[str, Any], bool]:
    """Run one scenario repeat, retrying on transient failures (server
    restarts, dropped connections, rate limits). Returns (row, was_retried)."""
    for attempt in range(retries + 1):
        try:
            return await _run_one_repeat(
                actae, sc, chat=chat, pipeline_model=pipeline_model,
                dry_run=dry_run, judge=judge, run_id=run_id, repeat_idx=repeat_idx,
            ), attempt > 0
        except Exception as exc:  # noqa: BLE001
            if not _is_transient(exc) or attempt == retries:
                raise
            await asyncio.sleep(min(10 * (2 ** attempt), 60))


# ---------------------------------------------------------------------------
# Suite driver
# ---------------------------------------------------------------------------


async def evaluate_langgraph_suite(
    actae: ActaeClient,
    scenarios: List[Scenario],
    *,
    chat: Any,
    pipeline_model: str,
    judge_model: str,
    dry_run: bool = False,
    judge_reps: int = 2,
    threshold: float = 7.0,
    tolerance: float = 1.5,
    repeats: int = 1,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every scenario through the LangGraph path and return the report
    dict: `SuiteReport.to_dict()` (from fork_fidelity_lib) PLUS
    `adapter: "langgraph"` and a per-scenario `langgraph` provenance block."""
    judge: Judge = (
        DeterministicJudge()
        if dry_run
        else LLMJudge(chat, model=judge_model, reps=judge_reps)
    )

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    results: List[ScenarioResult] = []
    langgraph_meta: Dict[str, Any] = {}

    for sc in scenarios:
        fork_cmps: List[FidelityComparison] = []
        rerun_cmps: List[FidelityComparison] = []
        repeat_rows: List[Dict[str, Any]] = []
        prefix_ok_all = True
        prefix_mismatches_all: List[str] = []
        channels_all: List[Dict[str, str]] = []
        total_calls = 0
        total_tokens = 0

        for i in range(repeats):
            row, retried = await _repeat_with_retry(
                actae, sc, chat=chat, pipeline_model=pipeline_model,
                dry_run=dry_run, judge=judge, run_id=run_id, repeat_idx=i,
            )
            prefix_ok_all = prefix_ok_all and row["prefix_ok"]
            prefix_mismatches_all.extend(row["prefix_mismatches"])
            channels_all.append(row["channels"])
            fork_cmps.append(row["fork"])
            rerun_cmps.append(row["rerun"])
            repeat_rows.append({
                "fork_fidelity": round(row["fork"].score, 2),
                "rerun_fidelity": round(row["rerun"].score, 2),
                "prefix_identical": row["prefix_ok"],
                "retried": retried,
                "channels": row["channels"],
                "observations": row["observations"],
                "fork_provenance": row["fork_provenance"],
                "langgraph": row["langgraph"],
            })
            total_calls += row["calls"]
            total_tokens += row["tokens"]
            langgraph_meta[sc.name] = row["langgraph"]

        fork_agg = aggregate_comparison(fork_cmps)
        rerun_agg = aggregate_comparison(rerun_cmps)
        passed, fail_reason = scenario_passed(
            prefix_ok_all, prefix_mismatches_all, fork_agg, rerun_agg,
            threshold=threshold, tolerance=tolerance,
        )
        result = ScenarioResult.from_parts(
            sc=sc,
            prefix_ok=prefix_ok_all,
            prefix_mismatches=prefix_mismatches_all,
            fork=fork_agg,
            rerun=rerun_agg,
            repeats=repeat_rows,
            calls=total_calls,
            tokens=total_tokens,
            channels=channels_all,
        )
        result.passed = passed
        result.fail_reason = fail_reason
        results.append(result)

    report = aggregate(
        results,
        threshold=threshold,
        tolerance=tolerance,
        dry_run=dry_run,
        pipeline_model=pipeline_model,
        judge_model=judge_model,
        judge_reps=judge_reps,
    )
    report_dict = report.to_dict()
    report_dict["adapter"] = ADAPTER
    for idx, r in enumerate(results):
        report_dict["scenarios"][idx]["langgraph"] = langgraph_meta.get(r.name)
    return report_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summary(report: Dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        f"Fidelity suite (adapter: langgraph) — LLM-judged fork vs no-fork re-run",
        f"generated_at: {report['generated_at']}   pipeline_model: {report['pipeline_model']}   "
        f"judge_model: {report['judge_model']} (temp {report['judge_temperature']}, reps {report['judge_reps']})",
        f"threshold: {report['threshold']}   tolerance: {report['tolerance']}   "
        f"dry_run: {report['dry_run']}   repeats: {len(report['scenarios'][0]['repeats']) if report['scenarios'] else 0}",
        "",
    ]
    for s in report["scenarios"]:
        status = "PASS" if s["pass"] else "FAIL"
        repeats_txt = ", ".join(
            f"fork={row['fork_fidelity']:.2f}/rerun={row['rerun_fidelity']:.2f}" for row in s["repeats"]
        )
        lines.append(
            f"  [{status}] {s['name']:<22} mean fork={s['fork_fidelity']['score']:>4.1f} "
            f"mean rerun={s['rerun_fidelity']['score']:>4.1f} prefix_identical={s['prefix_identical']}"
        )
        lines.append(f"          per-repeat: {repeats_txt}")
        if s.get("langgraph"):
            lg = s["langgraph"]
            lines.append(
                f"          fork: {lg['fork_channel']}  checkpoint={lg['forked_checkpoint_id'][:8]}  "
                f"restorable={lg['receipt']['restorable']}  "
                f"cursor {lg['receipt']['requested_cursor']}=={lg['receipt']['resolved_cursor']}  "
                f"sha256={lg['receipt']['source_state_sha256'][:12]}…"
            )
        if s["fail_reason"]:
            lines.append(f"          reason: {s['fail_reason']}")
        lines.append(f"          judge: {s['fork_fidelity']['verdict']} — {s['fork_fidelity']['rationale'][:140]}")
    lines.append("")
    lines.append(
        f"  OVERALL: {'PASS' if overall['pass'] else 'FAIL'}  {overall['passed']}/{overall['scenarios']} "
        f"scenarios  mean fork={overall['mean_fork_fidelity']}  mean rerun={overall['mean_rerun_fidelity']}  "
        f"worst fork={overall['worst_fork_fidelity']}  calls={overall['total_llm_calls']} tokens={overall['total_tokens']}"
    )
    return "\n".join(lines)


def _write_report(report: Dict[str, Any], report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"fork-fidelity-langgraph-{run_id}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (report_dir / "latest-langgraph.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path


async def main(args: argparse.Namespace) -> int:
    scenarios = (
        scenarios_by_name([s.strip() for s in args.scenarios.split(",")])
        if args.scenarios
        else ALL_SCENARIOS
    )

    chat = None
    if not args.dry_run:
        _load_deepseek_key()
        chat = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", timeout=180)

    actae = ActaeClient(api_key=ACTAE_API_KEY, endpoint=ACTAE_URL)
    await actae.connect()
    try:
        report = await evaluate_langgraph_suite(
            actae,
            scenarios,
            chat=chat,
            pipeline_model=args.model,
            judge_model=args.judge_model,
            dry_run=args.dry_run,
            judge_reps=args.judge_reps,
            threshold=args.threshold,
            tolerance=args.tolerance,
            repeats=args.repeats,
            run_id=args.run_id,
        )
    finally:
        await actae.disconnect()

    print(_summary(report))
    run_id = args.run_id or report["generated_at"].replace(":", "-").replace("T", "-")[:19]
    path = _write_report(report, Path(args.report_dir), run_id)
    print(f"\n  report: {path}")
    return 0 if report["overall"]["pass"] else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-judged LangGraph fork-fidelity suite (fork_thread vs no-fork re-run).",
    )
    p.add_argument("--dry-run", action="store_true", help="deterministic fake pipeline + judge; no API key (plumbing only)")
    p.add_argument("--scenarios", type=str, default=None, help="comma-separated scenario names (default: all)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="pipeline model (default: deepseek-chat)")
    p.add_argument("--judge-model", default=DEFAULT_MODEL, help="judge model (default: deepseek-chat)")
    p.add_argument("--judge-reps", type=int, default=2, help="judge samples averaged per comparison (default 2)")
    p.add_argument("--repeats", type=int, default=1, help="independent scenario runs averaged per scenario (default 1; 3+ for marketing evidence)")
    p.add_argument("--threshold", type=float, default=7.0, help="min fork fidelity to pass (default 7.0)")
    p.add_argument("--tolerance", type=float, default=1.5, help="max gap fork below re-run to pass (default 1.5)")
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="directory for JSON evidence reports")
    p.add_argument("--run-id", type=str, default=None, help="stable run id (default: timestamp)")
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if _args.judge_reps < 1 or _args.repeats < 1:
        raise SystemExit("--judge-reps and --repeats must be >= 1")
    sys.exit(asyncio.run(main(_args)))
