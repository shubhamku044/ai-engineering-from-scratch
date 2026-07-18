# The Learning Protocol

The rule that makes this repo worth more than the original: **I generate, I don't copy.**

## The per-lesson loop

For each lesson my mentor assigns:

1. **Read the concept, not the code.** Read the lesson's `docs/en.md`. Do **not** open the reference `code/` yet.
2. **Rebuild from blank.** Open an empty file in *this* repo and write the code myself. Get stuck — that's the point.
3. **Peek only when blocked.** Look at the reference just long enough to unblock the one thing, then close it and type it myself. Never copy-paste.
4. **Diff against the reference.** Compare mine to theirs. Every difference is a hole in my understanding — investigate the *why*.
5. **Break it & extend it.** Change something, predict the result before running. Do the lesson's exercises.
6. **Understanding check.** Answer my mentor's 6 questions or run the quiz. If I can't explain a line, I'm not done.

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
