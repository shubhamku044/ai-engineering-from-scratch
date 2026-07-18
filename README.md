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
| 01 | dev-environment | 🟡 | |
| 02 | git-and-collaboration | ⬜ | |
| 03 | gpu-setup-and-cloud | ⬜ | (adapt: MPS + Colab, no CUDA) |
| 04 | apis-and-keys | ⬜ | |
| 05 | jupyter-notebooks | ⬜ | |
| 06 | python-environments | ⬜ | |
| 07 | docker-for-ai | ⬜ | |
| 08 | editor-setup | ⬜ | |
| 09 | data-management | ⬜ | (already studied — will rebuild) |
| 10 | terminal-and-shell | ⬜ | |
| 11 | linux-for-ai | ⬜ | |
| 12 | debugging-and-profiling | ⬜ | |

*(Later phases added to this tracker as I reach them.)*

---

## Repo layout

```
phases/<phase>/<lesson>/
  ├── <my code>.py       # written from scratch
  └── notes.md           # the gotchas, the "why", what broke — in my own words
```

One commit per lesson. The commit history is the proof of work.
