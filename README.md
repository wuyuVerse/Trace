# TRACE — Recall-as-Execution

> Memory is not a photo album (store the past, retrieve the most-similar shot).
> Memory is a **ledger** (record every change, replay it to compute the present).

TRACE is the reference implementation of the **Recall-as-Execution** paradigm:
memory is an executable, typed, bitemporal ledger `L`, and reading it is a pure
function

```
execute(L, t, principal) -> S(t)
```

that folds the ledger up to wall-clock `t`, under the caller's principal, and
returns the currently-valid state `S(t)`. Membership in `S(t)` is a *theorem*
about the ledger — not a runtime similarity filter. Deleted or superseded facts
are provably unreachable; time-out-of-scope facts are provably unreachable;
counterfactual queries reduce to `execute(L\Δ, t, p)` with the same code path.

## Why it matters

Retrieval-augmented (RAG) memory answers "what did we say" by nearest-neighbour
lookup — which is silent about time, provenance, deletion, and consent. TRACE
answers "what is currently true" by folding a governed ledger, and can prove:

- **Temporal correctness.** If `φ` expired at `t` or was superseded before `t`,
  then `φ ∉ Recall(q,t,p)` for every query `q` and principal `p`.
- **Forgetting compliance.** If `DELETE(φ) ∈ L`, then no `q,t,p` can recall
  `φ` — deletion is structural, not a filter that can be bypassed.
- **Counterfactual separability.** If `Δ` changes the projection, the activated
  memory set changes; a working memory yields distinct answers for
  `L` and `L ⊕ Δ`.

Under `(State, ⊕, ∅)` the fold is a real monoid (associative + idempotent +
identity), so the same code that reconstructs one user's state is a legit
associative-scan over sharded ledgers.

## Quick start

```bash
pip install -e .
```

A minimal example — three ledger transitions, a governed recall, and a
counterfactual fork:

```python
from trace import TraceMemory
from trace.core import Transition

mem = TraceMemory(use_llm_compiler=False)   # or True to compile free text via LLM
mem.observe_transitions([
    Transition("k1", "config", "api_key", "ASSERT",     "2026-01", 1, content="KEY-A"),
    Transition("k2", "config", "api_key", "SUPERSEDE",  "2026-06", 2, content="KEY-B"),
    Transition("kd", "config", "api_key_old", "DELETE", "2026-06", 3, content="KEY-A"),
])

print(mem.recall(t="2026-07"))
# {'view': 'current_value',
#  'evidence': [{'subject':'config','attribute':'api_key','value':'KEY-B', ...}],
#  'excluded': [{'subject':'config','attribute':'api_key_old','reason':'deleted', ...}],
#  'admissible_ids': ['k2']}

print(mem.fork(remove_memory_ids={"k2"}, t="2026-07"))
# {'state': [{'subject':'config','attribute':'api_key','value':'KEY-A', 'source':'k1'}]}
```

`mem.check(memory_id, t, who)` gates whether a specific memory is admissible at
`(t, who)` — the action-layer guardrail for "did I just use a forgotten fact?"

## Directory guide

| Path | Contents |
|---|---|
| `trace/memory.py` | Top-level facade `TraceMemory` — observe / recall / fork / check / why |
| `trace/core/` | Deterministic execute engine — `machine`, `algebra`, `admissibility`, `governance`, `projections`, `relevance`, `certificate` |
| `trace/compile/` | Write-side: LLM compiler, deterministic compiler, write gate |
| `trace/verbalize/` | Read-side: Chain-of-Note verbaliser (governed state -> answer text) |
| `trace/runtime/` | OpenAI-compatible chat client (stdlib-only) |
| `trace/integrations/` | AMB benchmark adapter · MCP server · Codex hook · OpenCode plugin |
| `trace/experiments/` | Ablations, design-space & cost, cross-bench harnesses (LongMemEval / LoCoMo / MemBench / StructMemEval) |
| `trace/theory/` | Monoid laws, Z3 bounded verification, unit tests for the three theorems |

## Environment variables

The runtime reads a small set of environment variables. `TRACE_*` names are
preferred; `SRG_*` are accepted as aliases for internal backward compatibility.

| Variable | Default | Meaning |
|---|---|---|
| `TRACE_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible chat endpoint |
| `TRACE_LLM_API_KEY` (or `OPENAI_API_KEY`) | — | Bearer token for the endpoint |
| `TRACE_LLM_MODEL` | `gpt-4o-mini` | Compiler / verbaliser model |
| `TRACE_COMPILE_MAX_TOKENS` | `4096` | Cap for the LLM compiler round-trip |
| `TRACE_EMBED_MODEL` | `BAAI/bge-m3` | Sentence-Transformers path/id for dense recall |
| `TRACE_EMBED_DEVICE` | `cpu` | Device for the embedding model |
| `TRACE_LEDGER_PATH` | `~/.trace/ledger.jsonl` | Where the MCP server persists the ledger |
| `TRACE_LONGMEMEVAL_DATA` / `TRACE_LOCOMO_DATA` / `TRACE_MEMBENCH_DATA` / `TRACE_STRUCTMEM_GLOB` | see `trace/experiments/` | Data-file overrides for the cross-bench harnesses |

## Tests

```bash
python -m trace.theory.test_theorems       # three-theorem construction tests
python -m trace.theory.test_relevance      # deterministic read-relevance unit tests
python -m trace.theory.test_safety_govern  # sensitivity extraction + governance
python -m trace.theory.z3_bounded_verify   # Z3 bounded verification (requires z3-solver)
```

## Paper

The paradigm and experiments are described in the accompanying paper,
*TRACE: Recall-as-Execution — Memory as an Executable Ledger* (preprint TBD).

## License

MIT — see [`LICENSE`](LICENSE).

## Contact / issues

Source: <https://github.com/wuyuVerse/Trace>. Please file issues and PRs there.
