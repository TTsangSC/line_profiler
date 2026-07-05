# SPDX-License-Identifier: Apache-2.0
"""rollup / show — post-hoc passes over the ledger's MEASUREMENTS. Regenerable from the
ledger, so they report only measured quantities; inference-time/energy/carbon are left
for a dedicated modeling pass that reads these same measurements."""
from __future__ import annotations

import json
import os

from ._util import now_iso
from .gitutil import repo_root
from .ledger import ensure_data_dir, read_ledger, totals_path
from .schema import COMPACTION_KIND, SCHEMA


def cmd_rollup(args) -> None:
    rows = read_ledger()
    tot = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "billable_input": 0}
    by_model: dict[str, int] = {}
    by_activity: dict[str, int] = {}
    by_agent: dict[str, int] = {}
    turns = 0
    wall = 0.0
    web_search = web_fetch = 0
    commits = set()
    compaction = {"events": 0, "peak_context_tokens": 0, "summary_chars": 0}
    for r in rows:
        if r.get("kind") == COMPACTION_KIND:
            c = r.get("compaction", {})
            compaction["events"] += 1
            compaction["peak_context_tokens"] += c.get("peak_context_tokens", 0)
            compaction["summary_chars"] += c.get("summary_chars", 0)
            continue
        tk = r.get("tokens", {})
        for k in tot:
            tot[k] += tk.get(k, 0)
        turns += r.get("turns", 0)
        for m in r.get("models", []):
            by_model[m] = by_model.get(m, 0) + r.get("by_model", {}).get(m, {}).get("output", 0)
        act = r.get("activity") or "unlabeled"
        by_activity[act] = by_activity.get(act, 0) + tk.get("output", 0)
        agent = r.get("agent") or "unknown"
        by_agent[agent] = by_agent.get(agent, 0) + tk.get("output", 0)
        wall += r.get("time", {}).get("wall_clock_s") or 0
        web_search += r.get("server_tools", {}).get("web_search", 0)
        web_fetch += r.get("server_tools", {}).get("web_fetch", 0)
        c = r.get("commit") or ""
        if c and not c.startswith("pending@"):
            commits.add(c)
    totals = {
        "generated_at": now_iso(),
        "schema": SCHEMA,
        "ledger_rows": len(rows),
        "commits_accounted": len(commits),
        "turns": turns,
        "tokens": tot,
        "output_tokens_by_model": by_model,
        "output_tokens_by_activity": by_activity,
        "output_tokens_by_agent": by_agent,
        "server_tool_calls": {"web_search": web_search, "web_fetch": web_fetch},
        "time": {"wall_clock_s": round(wall, 1)},
        "compaction_signals": {  # measured; token/energy cost imputed post-hoc
            "events": compaction["events"],
            "peak_context_tokens": compaction["peak_context_tokens"],
            "summary_chars": compaction["summary_chars"],
        },
        "modeled_post_hoc": "inference_seconds, energy_kwh, carbon_gco2e — derived from "
                            "the measurements above by a separate pass; not stored here.",
    }
    ensure_data_dir()
    with open(totals_path(), "w", encoding="utf-8") as fh:
        json.dump(totals, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(totals, indent=2, ensure_ascii=False))
    print(f"# wrote {os.path.relpath(totals_path(), repo_root())}")


def cmd_show(args) -> None:
    rows = read_ledger()
    if not rows:
        print("ledger is empty.")
        return
    for r in rows:
        commit = (r.get("commit") or "")[:10]
        when = r.get("recorded_at", "")[:19]
        models = ",".join(r.get("models", []))
        if r.get("kind") == COMPACTION_KIND:
            c = r.get("compaction", {})
            print(f"{commit:12} {when} ~compaction {models:20} "
                  f"peak_ctx={c.get('peak_context_tokens', 0):>9} "
                  f"summary={c.get('summary_chars', 0):>6}c")
            continue
        tk = r.get("tokens", {})
        wall = r.get("time", {}).get("wall_clock_s")
        act = r.get("activity")
        print(f"{commit:12} {when}   measured   {models:20} "
              f"out={tk.get('output', 0):>7} billable_in={tk.get('billable_input', 0):>9} "
              f"wall={wall if wall is not None else '  -'}s"
              f"{('  <' + act + '>') if act else ''}")
