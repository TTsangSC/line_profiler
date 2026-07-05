# SPDX-License-Identifier: Apache-2.0
"""record / reconcile — attribute a session's measured turns to commits. Backend-agnostic:
turns and compaction events come from whichever `Backend` the CLI selected."""
from __future__ import annotations

import os

from ._util import now_iso, to_dt
from .backends import get_backend
from .config import registered_backends
from .gitutil import commit_meta, repo_root
from .ledger import (aggregate, append_row, base_row, compaction_row, read_ledger,
                     recorded_boundary_ts, session_watermark)
from .schema import COMPACTION_KIND


def record_compactions(backend, transcript, session_id, sha, commit_ts, lo_dt, hi_dt,
                       rows, activity, repo) -> int:
    """Append a reconstructed row for each compaction boundary in (lo_dt, hi_dt] not
    already recorded. hi_dt=None means unbounded (trailing sweep, for reconcile)."""
    seen = recorded_boundary_ts(rows, session_id)
    n = 0
    for ev in backend.parse_compaction_events(transcript):
        bts = ev["boundary_ts"]
        if bts in seen:
            continue
        bdt = to_dt(bts)
        if lo_dt is not None and bdt <= lo_dt:
            continue
        if hi_dt is not None and bdt > hi_dt:
            continue
        append_row(compaction_row(ev, sha, commit_ts, session_id, activity, repo, backend.name))
        n += 1
        print(f"  + compaction @ {bts}: peak_context~{ev['peak_context_tokens']:,} tok, "
              f"summary={ev['summary_chars']:,} chars [{ev['model']}] "
              f"(measured signals; token cost imputed post-hoc)")
    return n


def cmd_record(args) -> None:
    """Explicit path when the caller names a backend/session/transcript (single backend,
    normal discovery). Otherwise — the passive hook case, bare `record --commit HEAD` — walk
    the repo's registered backends and record whichever has a session matching this repo
    (strict; no global fallback), so Codex/mixed repos auto-record without a backend flag."""
    repo = os.path.basename(repo_root())
    if args.backend or args.transcript or args.session:
        backend = get_backend(args.backend)
        projects = args.projects_dir or backend.default_projects_dir()
        transcript = args.transcript or backend.find_transcript(projects, args.session)
        _record_transcript(backend, transcript, args, repo)
        return
    names = registered_backends()
    recorded = False
    for name in names:
        backend = get_backend(name)
        projects = args.projects_dir or backend.default_projects_dir()
        transcript = backend.find_transcript(projects, None, strict=True)
        if transcript:
            _record_transcript(backend, transcript, args, repo)
            recorded = True
    if not recorded:
        print(f"no matching session for any registered backend ({', '.join(names)}).")


def _record_transcript(backend, transcript, args, repo) -> None:
    session_id = os.path.splitext(os.path.basename(transcript))[0]
    sha, commit_ts = commit_meta(args.commit)
    rows = read_ledger()

    # Attribution window = (last watermark for this session, commit timestamp]. Bounding
    # the top at commit_ts (not "now") keeps work done AFTER this commit rolling forward
    # to the next commit's record instead of misattributing here.
    wm = session_watermark(rows, session_id)
    wm_dt, cut_dt = to_dt(wm), to_dt(commit_ts)

    measured_dup = (not args.force and any(
        r.get("session_id") == session_id and r.get("commit") == sha
        and r.get("kind") != COMPACTION_KIND for r in rows))
    if measured_dup:
        print(f"already recorded session {session_id[:8]} @ commit {sha[:8]} "
              f"(use --force to override); measured turns skipped.")
    else:
        new = [t for t in backend.parse_turns(transcript)
               if (wm_dt is None or to_dt(t["ts"]) > wm_dt) and to_dt(t["ts"]) <= cut_dt]
        if not new:
            print(f"no new turns for session {session_id[:8]} in "
                  f"({wm or 'epoch'}, {commit_ts}].")
        else:
            agg = aggregate(new)
            row = {**base_row(sha, commit_ts, session_id, args.label, repo, backend.name), **agg}
            append_row(row)
            tk = agg["tokens"]
            print(f"recorded {agg['turns']} turns for {sha[:8]} [{','.join(agg['models'])}]"
                  f"{(' <' + args.label + '>') if args.label else ''}: "
                  f"out={tk['output']} in={tk['input']} cache_w={tk['cache_write']} "
                  f"cache_r={tk['cache_read']}; wall={agg['time']['wall_clock_s']}s; "
                  f"inference-time/energy/carbon modeled post-hoc.")
            print(trailer_line(row))

    if not args.no_estimate_compaction:
        record_compactions(backend, transcript, session_id, sha, commit_ts,
                           wm_dt, cut_dt, rows, args.label, repo)


def cmd_reconcile(args) -> None:
    """Attribute any un-recorded trailing turns (per session) to a pending bucket, so a
    session that did work without committing is never dropped."""
    names = [args.backend] if args.backend else registered_backends()
    rows = read_ledger()
    repo = os.path.basename(repo_root())
    total = 0
    pending = f"pending@{now_iso()[:10]}"
    for name in names:
        backend = get_backend(name)
        projects = args.projects_dir or backend.default_projects_dir()
        for f in backend.session_transcripts(projects):
            sid = os.path.splitext(os.path.basename(f))[0]
            wm_dt = to_dt(session_watermark(rows, sid))
            new = [t for t in backend.parse_turns(f)
                   if wm_dt is None or to_dt(t["ts"]) > wm_dt]
            if new:
                agg = aggregate(new)
                row = {**base_row(pending, None, sid, args.label, repo, backend.name), **agg,
                       "note": "reconcile: un-committed turns swept so they are not undercounted"}
                append_row(row)
                total += agg["turns"]
                print(f"reconciled {agg['turns']} un-recorded turns for session {sid[:8]}"
                      f" [{backend.name}]{(' <' + args.label + '>') if args.label else ''}.")
            if not args.no_estimate_compaction:
                total += record_compactions(backend, f, sid, pending, None, wm_dt, None,
                                            rows, args.label, repo)
    if total == 0:
        print("nothing to reconcile; all session turns already accounted.")


def trailer_line(row: dict) -> str:
    """A git commit-trailer suggestion carrying the MEASURED usage (no modeled fields)."""
    tk = row["tokens"]
    return ("Resource-Usage: model={m}; tok in={i} cw={cw} cr={cr} out={o}; "
            "wall={w}s; inference-time/energy/carbon=modeled-post-hoc").format(
        m="+".join(row["models"]), i=tk["input"], cw=tk["cache_write"],
        cr=tk["cache_read"], o=tk["output"], w=row["time"]["wall_clock_s"])
