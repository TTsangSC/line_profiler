# SPDX-License-Identifier: Apache-2.0
"""The ledger: append-only, rolling JSONL shards of MEASURED usage, plus the readers
that de-duplicate them. Stores measurements only — inference-time, energy, and carbon
are derived post-hoc from these rows, so any modeling knob can change without re-recording.
"""
from __future__ import annotations

import glob
import json
import os

from ._util import now_iso, now_stamp, span_seconds
from .gitutil import repo_root
from .schema import COMPACTION_KIND, SCHEMA, TOKEN_KEYS, decode_row, encode_row

# Rotate a shard once it passes this size so no single JSONL file grows without bound.
MAX_LEDGER_BYTES = int(os.environ.get("LLM_RESOURCE_TALLY_MAX_LEDGER_BYTES", str(1_000_000)))


def data_dir() -> str:
    """`<host-repo-root>/.llm_resource_tally/`. Anchoring on the repo root (not this
    module) is the one rule that serves every install mode — vendored, submodule, pip —
    and keeps a submodule's ledger inside the submodule, committed to it not the parent."""
    return os.path.join(repo_root(), ".llm_resource_tally")


def ledger_dir() -> str:
    return os.path.join(data_dir(), "ledger")


def active_shard() -> str:
    return os.path.join(ledger_dir(), "ledger.jsonl")


def totals_path() -> str:
    return os.path.join(data_dir(), "lifetime-totals.json")


def shard_paths() -> list[str]:
    """All ledger shards, oldest first. Archives sort before the active `ledger.jsonl`
    (digits < 'j'); a pre-rolling flat `resource-ledger.jsonl` is read first for
    back-compat so newer shards win on de-dup."""
    paths = sorted(glob.glob(os.path.join(ledger_dir(), "*.jsonl")))
    legacy = os.path.join(data_dir(), "resource-ledger.jsonl")
    if os.path.exists(legacy):
        paths = [legacy] + paths
    return paths


def ensure_data_dir() -> str:
    """Create the data + ledger dirs and drop a self-contained `.gitattributes` marking
    the shards `merge=union` (so branches/rebases concatenate rows; readers de-dup)."""
    os.makedirs(ledger_dir(), exist_ok=True)
    ga = os.path.join(data_dir(), ".gitattributes")
    if not os.path.exists(ga):
        with open(ga, "w", encoding="utf-8") as fh:
            fh.write("# append-only ledger shards: keep rows from both sides on "
                     "merge/rebase;\n# readers de-duplicate by row identity.\n"
                     "ledger/*.jsonl merge=union\n")
    return data_dir()


def _row_identity(r: dict):
    """Stable identity independent of git SHAs surviving a rewrite; same identity = same
    observation, counted once."""
    sid = r.get("session_id")
    if r.get("kind") == COMPACTION_KIND:
        return ("compaction", sid, r.get("boundary_ts"))
    return ("measured", sid, r.get("commit"))


def read_ledger() -> list[dict]:
    """The ledger as a de-duplicated rich-row view (latest `recorded_at` wins per
    identity). Files stay pure append-only logs — safe to `merge=union` and to carry
    through a history rewrite — because readers collapse duplicates here."""
    order: list = []
    best: dict = {}
    for p in shard_paths():
        try:
            fh = open(p, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = decode_row(json.loads(line))
                except json.JSONDecodeError:
                    continue
                k = _row_identity(r)
                if k not in best:
                    order.append(k)
                cur = best.get(k)
                if cur is None or (r.get("recorded_at") or "") >= (cur.get("recorded_at") or ""):
                    best[k] = r
    return [best[k] for k in order]


def _maybe_rotate() -> None:
    """Best-effort: if the active shard is already at/over the size cap, archive it to a
    timestamped name so appends start a fresh file. os.replace is atomic; concurrent
    rotators just race harmlessly (the loser's getsize raises and is ignored)."""
    active = active_shard()
    try:
        if os.path.getsize(active) >= MAX_LEDGER_BYTES:
            arch = os.path.join(ledger_dir(), f"ledger.{now_stamp()}.jsonl")
            if not os.path.exists(arch):
                os.replace(active, arch)
    except OSError:
        pass


def append_row(row: dict) -> None:
    """Encode to the compact schema and append one line under an flock (so concurrent
    agents don't interleave), rotating the shard first if it has grown too large."""
    import fcntl
    ensure_data_dir()
    _maybe_rotate()
    line = json.dumps(encode_row(row), separators=(",", ":"), ensure_ascii=False) + "\n"
    with open(active_shard(), "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(line)
        fcntl.flock(fh, fcntl.LOCK_UN)


def session_watermark(rows: list[dict], session_id: str) -> str:
    """Max turn timestamp already recorded for this session ('' if none)."""
    hi = ""
    for r in rows:
        if r.get("session_id") == session_id:
            rng = r.get("turn_ts_range") or [None, None]
            if rng[1] and rng[1] > hi:
                hi = rng[1]
    return hi


def recorded_boundary_ts(rows: list[dict], session_id: str) -> set:
    """Boundary timestamps of compaction estimates already recorded for a session
    (dedup key so re-running never double-counts a compaction event)."""
    return {r.get("boundary_ts") for r in rows
            if r.get("session_id") == session_id and r.get("kind") == COMPACTION_KIND}


def aggregate(turns: list[dict]) -> dict:
    """Sum the MEASURED usage of a set of turns. Wall-clock and the timestamp range are
    measured; inference-time/energy/carbon are derived post-hoc."""
    tok = {k: 0 for k in TOKEN_KEYS}
    by_model: dict[str, dict] = {}
    web_search = web_fetch = 0
    for t in turns:
        for k in TOKEN_KEYS:
            tok[k] += t["usage"][k]
        bm = by_model.setdefault(t["model"], {k: 0 for k in TOKEN_KEYS})
        for k in TOKEN_KEYS:
            bm[k] += t["usage"][k]
        web_search += t["web_search"]
        web_fetch += t["web_fetch"]
    ts_lo = turns[0]["ts"] if turns else None
    ts_hi = turns[-1]["ts"] if turns else None
    return {
        "turns": len(turns),
        "models": sorted(by_model),
        "tokens": {
            "input": tok["input_tokens"],
            "cache_write": tok["cache_creation_input_tokens"],
            "cache_read": tok["cache_read_input_tokens"],
            "output": tok["output_tokens"],
            "billable_input": (tok["input_tokens"]
                               + tok["cache_creation_input_tokens"]
                               + tok["cache_read_input_tokens"]),
        },
        "by_model": {m: {"input": v["input_tokens"],
                         "cache_write": v["cache_creation_input_tokens"],
                         "cache_read": v["cache_read_input_tokens"],
                         "output": v["output_tokens"]}
                     for m, v in by_model.items()},
        "server_tools": {"web_search": web_search, "web_fetch": web_fetch},
        "time": {"wall_clock_s": span_seconds(ts_lo, ts_hi)},
        "turn_ts_range": [ts_lo, ts_hi],
    }


def base_row(sha: str, commit_ts, session_id: str, activity, repo: str,
             agent: str) -> dict:
    """Common identity/provenance fields shared by measured and compaction rows. `agent`
    is the backend that produced the row (e.g. "claude-code", "codex") so a repo can mix
    backends in one ledger."""
    return {"schema": SCHEMA, "recorded_at": now_iso(), "repo": repo,
            "commit": sha, "commit_ts": commit_ts, "agent": agent,
            "activity": activity, "session_id": session_id}


def compaction_row(ev: dict, sha: str, commit_ts, session_id: str,
                   activity, repo: str, agent: str) -> dict:
    """A compaction row records only MEASURED signals — no fabricated token counts."""
    row = base_row(sha, commit_ts, session_id, activity, repo, agent)
    row.update(kind=COMPACTION_KIND, source="reconstructed",
               boundary_ts=ev["boundary_ts"], models=[ev["model"]],
               compaction={"peak_context_tokens": ev["peak_context_tokens"],
                           "summary_chars": ev["summary_chars"]})
    return row
