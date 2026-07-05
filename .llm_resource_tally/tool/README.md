# llm_resource_tally — measured LLM resource accounting (per commit)

A small, **self-contained** tool (Python **stdlib only** — zero dependencies) that records
the compute each commit cost an LLM agent, so a repo's lifetime *resource utilization* can
be estimated. Token counts and model are **measured** verbatim from the agent's own session
transcript; energy and carbon are derived later from the recorded tokens/time and the commit
timestamp (which fixes the grid's carbon intensity at that moment).

Everything the tool owns lives under **one** directory, **`.llm_resource_tally/`** in your repo
root, committed alongside your code: the ledger data in `ledger/`, and the vendored tool code in
`tool/`.

## Quick start

From inside the repo you want to track:

```bash
curl -fsSL https://raw.githubusercontent.com/Erotemic/llm_resource_tally/main/install.sh | sh
```

That vendors a self-contained copy of the tool into `.llm_resource_tally/tool/` and wires a git
`post-commit` hook (plus a managed `AGENTS.md` block) — offline after the initial fetch. Review
and **commit `.llm_resource_tally/` + `AGENTS.md`** to share it. From then on every `git commit`
auto-records what it cost.

**Claude Code users** — add precise cross-repo attribution (recommended):
```bash
python3 .llm_resource_tally/tool install --claude   # also wires a Claude PostToolUse hook
```

Prefer pip or a git submodule, or want to vendor elsewhere / re-wire a fresh clone / update /
uninstall? See **[docs/install.md](docs/install.md)**.

## Usage

With the hook installed, recording is automatic. `<rt>` below is `python3 .llm_resource_tally/tool`:

```bash
<rt> reconcile --label review   # sweep turns that produced no commit (planning, chat, review)
<rt> rollup                     # refresh lifetime totals -> .llm_resource_tally/lifetime-totals.json
<rt> show                       # print the ledger
```

The one habit to keep: **at session end, run `<rt> reconcile && <rt> rollup`** — the hook only
fires on commits, so `reconcile` is what captures planning/chat/review that produced none. Tag
work with `--label` (e.g. `record --label implementation`) so `rollup` can break usage down
`by_activity`. Codex agents can record with `<rt> record --backend codex`; other non-Claude
agents use `<rt> record --backend <name> --transcript <session.jsonl>`.

## How tracking works

The tool reads the **session transcript** your agent already writes (Claude Code and Codex both
do) and, per **turn** (one API call), keeps only the measurements the agent itself logged — token
counts, model, timestamps — **never message content, code, or prompts**. Each turn is attributed
to the commit it feeds; turns that produce no commit are swept by `reconcile`. Rows are deduped by
message id and appended to the committed ledger.

- **Measured & stored** (verbatim from the transcript): model; tokens by kind (input, cache-write,
  cache-read, output); server-tool calls where the agent reports them; turn timestamps +
  wall-clock; and context-compaction signals (peak context, summary size) when the agent compacts.
- **Derived later, never stored:** inference-seconds, energy (kWh), carbon (gCO₂e), USD. Each is
  an assumption *over* the measurements, so the modeling pass can change without re-recording.
- **Not captured** (nothing to read): a commit made with no agent session; usage older than the
  agent's transcript retention (Claude Code defaults to 30 days); and a session in one repo that
  commits into another, which needs a hint to attribute (the Claude `--claude` hook, or a one-line
  manual bridge).

**Does the agent have to think about this?** For Claude Code, no — the git `post-commit` hook
records every commit automatically. Codex (or a mix) is the same after a one-time `install
--backend codex`, which registers it in `.llm_resource_tally/settings.json` so the hook records it
too; the hook walks the registered backends and records whichever agent actually produced the
commit (matched strictly to this repo, so an unrelated session is never mis-attributed). The only
manual step is the **session-end sweep** (`reconcile && rollup`) that captures non-committing
turns, since a post-commit hook can't fire without a commit; the managed `AGENTS.md` block reminds
the agent to run it.

Case-by-case details — cross-repo, submodules, non-committing work, history rewrites, compaction,
per-backend field mapping, and the exact on-disk fields — are in the docs below.

## Documentation

- **[Install & wiring](docs/install.md)** — every install route (curl / pip / submodule), the
  `<rt>` alias, hook-mode options, update/uninstall, and self-replicating installs.
- **[Attribution](docs/attribution.md)** — how cost is attributed to commits, cross-repo and
  submodule cases, the Claude `--claude` hook, correctness guarantees, history rewrites, and
  context compaction.
- **[Data model](docs/data-model.md)** — where data lives, the measurements-only principle, the
  compact rolling ledger, and `lifetime-totals.json`.
- **[Backfill](docs/backfill.md)** — recovering usage from before the hook was installed, and the
  retention horizon that bounds how far back you can go.
- **[Backends](docs/backends.md)** — the agent-agnostic core and how to add one (Codex, etc.).
- **[Development](docs/development.md)** — package layout, the three invocation styles, tests & CI.
- **[Related work](docs/related-work.md)** — how this differs from ccusage, claude-budget,
  llm-usage-metrics, Claude Code Analytics, and live monitors.

## License

Apache-2.0. See [LICENSE](LICENSE).
