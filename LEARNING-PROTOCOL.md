# The Learning Protocol

The rule that makes this repo worth more than the original: **I generate, I don't copy.**

## The per-lesson loop

For each lesson my mentor assigns:

1. **Read the concept, not the code.** Read the lesson's `docs/en.md`. Do **not** open the reference `code/` yet.
2. **Rebuild from blank.** My mentor creates the empty target file (just a comment header stating the goal). I write the code myself from there. Get stuck — that's the point.
3. **Peek only when blocked.** Look at the reference just long enough to unblock the one thing, then close it and type it myself. Never copy-paste.
4. **Diff against the reference.** Compare mine to theirs. Every difference is a hole in my understanding — investigate the *why*.
5. **Break it & extend it.** Change something, predict the result before running. Do the lesson's exercises.
6. **Understanding check.** Answer my mentor's questions or run the quiz. If I can't explain a line, I'm not done.
7. **Notes get written up.** My mentor drafts `notes.md` — but *only* from the explanations I gave out loud in the session. It's a transcript of my own understanding, not new content. I read and revise later.

## The one line that can't move

**I generate all the code myself, and I explain every concept out loud myself.** The mentor may write up `notes.md`, run commands, and commit — but if the mentor ever writes an explanation for something I never explained, we've slid back into copy-paste with the mentor as the source. Notes are a *record* of my understanding, never a *substitute* for it.

## Commits

- **One commit per lesson** — the history is the proof of work.
- **No AI co-author trailer.** Commits do **not** include a `Co-Authored-By: Claude` line. This is my work; the commits say so.

## Definition of "done" for a lesson

I can say **yes** to all three:
- [ ] I rebuilt the core on a blank screen without looking.
- [ ] I can explain every line out loud.
- [ ] I passed the check (6 Qs or quiz).

Only then does my mentor unlock the next lesson.

## Time budget per lesson

~20% reading concept · ~70% writing/breaking code · ~10% checking understanding.
If it flips to mostly-reading, I'm watching a course, not taking one.

## Mac / no-CUDA rules

- Local GPU = **MPS**: `torch.backends.mps.is_available()`, use `device = "mps"`.
- CUDA-only lessons → run on **Colab** or a cloud GPU. Note it in `notes.md`, don't fake it.
- Pin **Python 3.12** via `uv` for every environment (ML wheels don't support 3.14 yet).
