# AI Engineering — My Build

My from-scratch rebuild of the *AI Engineering from Scratch* curriculum.
I write every line myself. The reference repo is the answer key I check against **after** I try.

**Machine:** MacBook, Apple M4 Pro (arm64), macOS 26.
**No NVIDIA/CUDA** — I use Apple's **MPS** (Metal) GPU backend for local work, and cloud GPUs (Colab / cloud instances) for training-heavy lessons.
**Python:** pinned to **3.12** (system 3.14 is too new for the ML stack).

---

## How this works (the protocol)

For every lesson, my mentor gives me tasks. I do not move to the next lesson until I can:
1. **Rebuild the core code from a blank file** — no peeking.
2. **Explain every line** to another person.
3. **Pass the understanding check** (6 questions or the quiz).

See [LEARNING-PROTOCOL.md](LEARNING-PROTOCOL.md) for the full loop.

> Reading code is not learning. Generating it from scratch is. Recognition is worthless; generation is the skill.

---

## Focused path (not all 503 lessons)

The full course is 503 lessons. I'm taking the **employable path first**, then circling back:

**Phase 0** (setup) → **Phase 1** (math, enough to not be lost) → **Phase 2–3** (ML + DL core) →
**Phase 11** (LLM engineering: RAG, fine-tuning, evals) → **Phase 13** (tools/protocols) → **Phase 14** (agents).

Everything else (CV, speech, RL, multimodal) is optional depth I add when a goal needs it.

---

## Progress tracker

Legend: ⬜ not started · 🟡 in progress · ✅ done (rebuilt from scratch + explained + quiz passed)

### Phase 00 — Setup & Tooling
| # | Lesson | Status | Notes |
|---|--------|--------|-------|
| 01 | dev-environment | ✅ | MPS-aware verify.py; rebuilt from blank |
| 02 | git-and-collaboration | ✅ | proven live: committed + pushed this session |
| 03 | gpu-setup-and-cloud | ✅ | MPS detection covered by verify.py; benchmark as extension |
| 04 | apis-and-keys | ✅ | first OpenAI call: SDK + raw HTTP, tokens, key guard |
| 05 | jupyter-notebooks | ✅ | confirmed: restart-and-run-all habit, hidden-state trap |
| 06 | python-environments | ✅ | proven live: uv venv + pyproject + lockfile |
| 07 | docker-for-ai | ✅ | confirmed: image = full OS env, not just deps |
| 08 | editor-setup | ✅ | personal choice |
| 09 | data-management | ✅ | confirmed: DVC/git-LFS/S3 for big data, not git |
| 10 | terminal-and-shell | ✅ | proven live: zsh throughout |
| 11 | linux-for-ai | ✅ | confirmed: OOM killer / dmesg diagnosis |
| 12 | debugging-and-profiling | ✅ | confirmed: profiler vs time.time(), cProfile |

### Phase 01 — Math Foundations
| # | Lesson | Status | Notes |
|---|--------|--------|-------|
| 01 | linear-algebra-intuition | ✅ | Vector/Matrix/projection/Gram-Schmidt from scratch, self-tested |
| 02 | vectors-matrices-operations | ✅ | element-wise ops, determinant, inverse, dense layer — from scratch |
| 03 | matrix-transformations | ✅ | transforms, composition, eigenvalues/vectors — extends OperableMatrix (scaffold mode) |
| 04 | calculus-for-ml | ✅ | derivatives, gradient, gradient descent, trained linear regression from scratch |
| 05 | chain-rule-and-autodiff | ⬜ | spine: this IS backprop — do it properly |
| 06 | probability-and-distributions | ⬜ | spine: losses, likelihood, softmax |
| 08 | optimization | ⬜ | spine: SGD / momentum / Adam |

**Lean-spine plan:** doing only the load-bearing math (03→04→05→06→08) then jumping to
Phase 2. Deferred until a project needs them: 07, 09–22 (info theory, SVD, tensors,
numerical stability, sampling, convex opt, complex/Fourier, graphs, stochastic processes).

---

## Repo layout

```
phases/<phase>/<lesson>/
  ├── <my code>.py       # written from scratch (always mine)
  └── notes.md           # mentor writes it up from what I explained out loud; I revise later
```

One commit per lesson. The commit history is the proof of work.
