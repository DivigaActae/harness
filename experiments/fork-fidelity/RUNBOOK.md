# Fork-fidelity test runbook (2026-08-12)

A faithful, copy-paste record of the commands used to prove fork/resume
fidelity across all three SDKs and every framework adapter. The evidence it
produces is documented in `docs/FORK_FIDELITY.md` (including the committed
reports in `examples/fidelity-reports/`).

Everything here targets the **dev (embedded) Actae** — no Docker, no external
Postgres.

---

## 1. Prerequisites

```bash
# Python SDK (editable)
pip install -e ./sdks/python

# TypeScript SDK deps
cd sdks/typescript && npm install

# Go SDK — no install step (module lives at sdks/go)
```

Environment (all suites read these):

```bash
export ACTAE_URL=http://localhost:8002
export ACTAE_API_KEY=sk-dev-0000000000000000000000   # dev-mode well-known key
```

The real-LLM fidelity engines need a DeepSeek key. They read it from
`actae/.env` (`DEEPSEEK_API_KEY=…`) automatically, or from the environment.
Dry-run modes need no key at all.

## 2. Start the dev server

```bash
cd actae
# Raise the auth rate limits: the heavy suites blow through the default
# 5/s burst-20 limiter and get 429s (and a 300s IP lockout).
ACTAE_IP_AUTH_RATE=1000 ACTAE_IP_AUTH_CAPACITY=2000 cargo run
```

That binds HTTP on :8002 and embedded Postgres on :5546 with a data dir of
`~/.actae/data`.

### Isolated instance (for long LLM runs)

If another process keeps managing/restarting the :8002 instance (e.g. a
concurrent lifecycle-test workstream), run a dedicated instance on its own
port + data dir so long suites can't be interrupted:

```bash
cd actae
ACTAE_PORT=8032 ACTAE_EMBEDDED_PG_PORT=5648 \
  ACTAE_DATA_DIR=/tmp/actae-fid-8032 \
  ACTAE_PG_BIN_DIR=~/.actae/data/pg-bin/*-darwin-arm64v8/bin \
  ACTAE_IP_AUTH_RATE=5000 ACTAE_IP_AUTH_CAPACITY=10000 \
  ./target/debug/actae
# then export ACTAE_URL=http://localhost:8032
```

`ACTAE_PG_BIN_DIR` reuses the already-downloaded Postgres binaries and skips
the one-time ~15-60 MB download; on a fresh checkout it can be omitted (it
will download once).

## 3. Python SDK

```bash
# Unit tests (no server)
python3 -m pytest sdks/python/tests/

# Deep deterministic live suite — byte-identical full-vs-fork proof,
# modified-prompt fork differs only at the refined step, fork-at-every-step,
# fork-of-fork, lineage_only, contamination guards, Claude fork_session,
# StateManager.fork, Codex mirror. Needs a running Actae.
python3 -m pytest sdks/python/tests/test_fork_resume_deep.py -v

# Framework-adapter live forks: CrewAI / LangChain / OpenAI-Agents
# (test_adapter_fork_live.py); LangGraph (test_langgraph_live.py)
python3 -m pytest sdks/python/tests/test_adapter_fork_live.py -v
python3 -m pytest sdks/python/tests/test_langgraph_live.py -v

# Fidelity infra unit tests (no server, no LLM)
python3 -m pytest sdks/example-tests/python/test_fork_fidelity_lib.py

# Fidelity suite live: dry-run plumbing (needs only Actae) + real judge
# (needs DEEPSEEK_API_KEY)
python3 -m pytest sdks/example-tests/python/test_fork_fidelity_suite_live.py -v
```

## 4. Python LLM-judged fidelity examples

```bash
cd examples

# Plumbing only — deterministic pipeline + judge, no API key:
python3 fork_fidelity_suite.py --dry-run
python3 fork_fidelity_langgraph.py --dry-run

# Real evidence — the marketing-grade bar (DeepSeek from actae/.env;
# ~10-15 min each, real tokens):
python3 fork_fidelity_suite.py --repeats 3 --judge-reps 1
python3 fork_fidelity_langgraph.py --repeats 3 --judge-reps 1

# Subset / tuning (note: the --scenarios comma-split bug was fixed 2026-08-12):
python3 fork_fidelity_suite.py --scenarios s3-localization,s5-product-brief --repeats 2 --judge-reps 2

# Exit code is 0 (all scenarios passed) or 1.
# Reports: examples/fidelity-reports/fork-fidelity-<ts>.json /
#          fork-fidelity-langgraph-<ts>.json (+ latest.json pointers)
```

`fork_fidelity_suite.py` = the original manual/`AgentSession` path;
`fork_fidelity_langgraph.py` = the same five scenarios through a real
`StateGraph` compiled with `ActaeCheckpointSaver`, forking via
`fork_thread` at an intermediate checkpoint.

## 5. Go SDK

```bash
cd sdks/go
go build ./... && go vet ./... && gofmt -l .
go test ./... -count=1        # unit (no server)
go test -race ./...           # race detector

# Deep deterministic live suite (15 cases — byte-identical full-vs-fork,
# modified-prompt, edge cases). Needs a running Actae.
go test ./ -run TestForkDeepLive -v

# Live smoke (record/replay/fork/session/WS/StateManager)
go run ./smoke/smoke.go

# LLM-judged fidelity suite (Go port of fork_fidelity_suite.py)
cd examples/fidelity
go run . --dry-run                          # plumbing, no API key
go run . --repeats 3 --judge-reps 1         # real evidence (~10-15 min)
go run . --scenarios s3-localization,s5-product-brief --repeats 2
# Reports: examples/fidelity-reports/fork-fidelity-go-<ts>.json + latest-go.json
```

## 6. TypeScript SDK

```bash
cd sdks/typescript
npm run typecheck && npm run build          # build is required for examples
npm test                                    # unit (no server)

# Deep deterministic live suite (15 cases) — needs a running Actae.
# NOTE: vitest.live.config.ts defaults ACTAE_URL to :8002; override it when
# testing a different instance.
ACTAE_LIVE=1 npx vitest run --config vitest.live.config.ts test/fork_deep_live.test.ts

# Framework-adapter live forks (LangGraph forkThread, Claude forkSession,
# LangChain fork, Copilot fork)
ACTAE_LIVE=1 npx vitest run --config vitest.live.config.ts test/adapters_fork_live.test.ts

# Full live suite (the above + record/replay/fork, WS, session)
ACTAE_LIVE=1 npm run test:live

# LLM-judged fidelity (single file, imports the built dist):
# session adapter (AgentSession) and langgraph adapter (real @langchain/langgraph)
node examples/fidelity/index.ts --adapter session --dry-run
node examples/fidelity/index.ts --adapter langgraph --dry-run
node examples/fidelity/index.ts --adapter session --repeats 3 --judge-reps 1
node examples/fidelity/index.ts --adapter langgraph --repeats 3 --judge-reps 1
# Reports: examples/fidelity-reports/fork-fidelity-ts-<ts>.json /
#          fork-fidelity-langgraph-ts-<ts>.json (+ latest-ts.json pointers)
```

## 7. What the suites proved (results on 2026-08-12)

| Suite | result |
|-------|--------|
| Python unit | 533 passed, 35 skipped |
| Python deep live + adapter fork live | 7 passed |
| Go unit | passed (incl. `TestForkStateFnNotWrapped`, `TestStateManagerFork`) |
| Go deep live (`TestForkDeepLive`) | 15/15 passed |
| TS unit | 435 passed, 22 skipped; typecheck + build clean |
| TS deep live + adapter fork live | 19/19 passed |

LLM-judged fidelity (real DeepSeek), **5/5 scenarios PASS in every run**:

| Suite | Adapter | repeats | mean fork | mean re-run | worst fork |
|-------|---------|---------|-----------|-------------|------------|
| Python | `AgentSession` | 3 | 8.47 | 7.07 | 7.67 |
| Python | LangGraph `fork_thread` | 3 | 8.40 | 6.87 | 7.33 |
| Go | `AgentSession`/`Resume` | 3 | 8.20 | 7.67 | 7.67 |
| TS | `AgentSession` | 1 | 8.80 | 8.20 | 8.00 |
| TS | LangGraph `forkThread` | 1 | 8.40 | 7.80 | 8.00 |

Every fork in every suite reported `prefix_identical=true` (inherited prefix
byte-identical to baseline), a restorable exact boundary (requested ==
resolved cursor), and a non-empty source-state SHA-256.

## 8. Gotchas hit while running (so you don't re-hit them)

- **Rate limits**: default HTTP auth limiter is 5/s burst 20. The deep and
  fidelity suites need `ACTAE_IP_AUTH_RATE`/`ACTAE_IP_AUTH_CAPACITY` raised
  at server start, or you get `Rate limited` errors and eventually a 300s
  IP lockout.
- **TS vitest URL**: `vitest.live.config.ts` sets `ACTAE_URL` to
  `http://localhost:8002` unless overridden — the TS live tests will silently
  fail against a different port if you forget the env var.
- **DEEPSEEK_API_KEY**: all three fidelity engines load it from `actae/.env`
  if not in the environment; `--dry-run` never needs it.
- **`--scenarios` flag**: the Python reference suite previously iterated the
  comma string char-by-char (a bug); it was fixed to split on commas. The Go
  and TS ports always did it correctly.
- **Parallel workstreams**: a concurrent process managing managed storage
  instances will kill long-running servers (observed several times on
  2026-08-12). Use the isolated-instance recipe (§2) for LLM runs.
- **TS reports at repeats=1**: the two TS evidence runs were captured at
  `--repeats 1` because the shared server kept being killed before a 3-repeat
  run could finish. Re-run at `--repeats 3` on a stable server to match the
  Python/Go statistical bar.

## 9. Bugs this runbook's suites caught (all fixed)

1. **Go/TS fork `stateFn` wrap parity bug** — forks never persisted their own
   evolving steps; nested forks inherited only the prefix. Fixed in
   `sdks/go/session.go` and `sdks/typescript/src/session.ts`; parity tests:
   Go `TestForkStateFnNotWrapped`, TS `does NOT wrap stateFn …`.
2. **Server 500** forking an event-less channel with ≥2 state snapshots —
   `actae/src/events/store.rs` probe query missing `LIMIT 1`.
3. **Python `--scenarios` comma-split bug** — `examples/fork_fidelity_suite.py`.
