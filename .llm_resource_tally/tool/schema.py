# SPDX-License-Identifier: Apache-2.0
"""On-disk ledger schema (compact) <-> in-memory row (rich).

The ledger file stores MEASUREMENTS only — no modeling assumptions. On disk we use a
COMPACT representation to keep the JSONL small: short keys, positional token arrays, and
omitted-when-empty optional fields, serialized with no whitespace. The rest of the code
works with a RICH dict (readable keys); this module is the only place that knows the
compact form. lifetime-totals.json keeps fully readable keys.

Compact row (`v:3`) key map
---------------------------
  v    schema version (3)                 rec  recorded_at (ISO)
  r    repo basename                      c    commit sha (or "pending@YYYY-MM-DD")
  ct   commit committer-date (ISO|null)   a    agent (e.g. "claude-code")
  sid  session id                         act  activity label (omitted if null)
  m    models (list)
  measured rows:
    n   turns                             t    tokens [input, cache_write, cache_read, output]
    bm  by_model {model: [i,cw,cr,out]}   st   server tools [web_search, web_fetch] (omit if 0,0)
    w   wall_clock_s (float|null)         tr   turn timestamp range [lo, hi]
  compaction rows (k="cx"):
    bt  boundary timestamp                cp   [peak_context_tokens, summary_chars]

billable_input is DERIVED (input+cache_write+cache_read), never stored.
"""
from __future__ import annotations

SCHEMA_VERSION = 3
SCHEMA = f"llm-resource-tally/v{SCHEMA_VERSION}"

# Context-compaction: a real LLM call the harness performs but logs NO usage object for
# (only a compact_boundary marker + isCompactSummary text). We store only the measured
# signals; kind marks the row so the rollup tallies it separately.
COMPACTION_KIND = "compaction-estimate"

# Transcript usage keys, in the canonical order used by the positional `t` array.
TOKEN_KEYS = ("input_tokens", "cache_creation_input_tokens",
              "cache_read_input_tokens", "output_tokens")


def _tok_list(tok: dict) -> list:
    return [tok.get("input", 0), tok.get("cache_write", 0),
            tok.get("cache_read", 0), tok.get("output", 0)]


def encode_row(r: dict) -> dict:
    """Rich row -> compact on-disk row."""
    out = {"v": SCHEMA_VERSION, "rec": r.get("recorded_at"), "r": r.get("repo"),
           "c": r.get("commit"), "ct": r.get("commit_ts"), "a": r.get("agent"),
           "sid": r.get("session_id"), "m": r.get("models", [])}
    if r.get("activity") is not None:
        out["act"] = r["activity"]
    if r.get("kind") == COMPACTION_KIND:
        out["k"] = "cx"
        out["bt"] = r.get("boundary_ts")
        cp = r.get("compaction", {})
        out["cp"] = [cp.get("peak_context_tokens", 0), cp.get("summary_chars", 0)]
        return out
    out["n"] = r.get("turns", 0)
    out["t"] = _tok_list(r.get("tokens", {}))
    bm = r.get("by_model", {})
    if bm:
        out["bm"] = {m: _tok_list(v) for m, v in bm.items()}
    st = r.get("server_tools", {})
    ws, wf = st.get("web_search", 0), st.get("web_fetch", 0)
    if ws or wf:
        out["st"] = [ws, wf]
    out["w"] = (r.get("time") or {}).get("wall_clock_s")
    out["tr"] = r.get("turn_ts_range", [None, None])
    return out


def _tok_dict(a: list) -> dict:
    a = (list(a) + [0, 0, 0, 0])[:4]
    return {"input": a[0], "cache_write": a[1], "cache_read": a[2], "output": a[3]}


def decode_row(d: dict) -> dict:
    """Compact (or legacy verbose) on-disk row -> rich in-memory row.

    Legacy pre-v3 rows carried a `tokens` object / `schema` string already in rich form;
    those pass through unchanged, so an older ledger still reads."""
    if "tokens" in d or "schema" in d or "v" not in d:
        return d  # legacy verbose row — already rich
    rich = {"schema": SCHEMA, "recorded_at": d.get("rec"), "repo": d.get("r"),
            "commit": d.get("c"), "commit_ts": d.get("ct"), "agent": d.get("a"),
            "activity": d.get("act"), "session_id": d.get("sid"),
            "models": d.get("m", [])}
    if d.get("k") == "cx":
        cp = d.get("cp", [0, 0])
        rich.update(kind=COMPACTION_KIND, source="reconstructed",
                    boundary_ts=d.get("bt"),
                    compaction={"peak_context_tokens": cp[0], "summary_chars": cp[1]})
        return rich
    tok = _tok_dict(d.get("t", []))
    tok["billable_input"] = tok["input"] + tok["cache_write"] + tok["cache_read"]
    st = (d.get("st") or [0, 0])
    rich.update(turns=d.get("n", 0), tokens=tok,
                by_model={m: _tok_dict(v) for m, v in (d.get("bm") or {}).items()},
                server_tools={"web_search": st[0], "web_fetch": st[1]},
                time={"wall_clock_s": d.get("w")},
                turn_ts_range=d.get("tr", [None, None]))
    return rich
