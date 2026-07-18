# Lesson 01 — Dev Environment · notes

> My own words, drafted from how I explained each concept while building this.
> Revise freely — this is mine to sharpen.

## What this lesson produced
`verify.py` — a script that prints PASS/FAIL for the tools and libs this course
actually needs *on this machine* (Python 3.12+, git/node/cargo, numpy/torch), and
reports the real torch device string. It's lesson 01 because a broken environment
turns every later lesson into a fight with tooling instead of learning.

## The MPS vs CUDA thing (the real lesson)
`check_device()` only checked for MPS at first. On a Linux box with an NVIDIA GPU,
`torch.backends.mps.is_available()` is `False`, so it would wrongly fall back to
`"cpu"` even though CUDA was available. MPS is Apple's Metal backend; `torch.cuda.*`
is NVIDIA-only and is `False` forever on my M4. Fix = check backends in priority order:

```python
if torch.backends.mps.is_available():
    return True, "mps"
if torch.cuda.is_available():
    return True, "cuda"
return True, "cpu"
```

The returned string is exactly what I pass to `.to(device)` on tensors/models later.

## which() vs subprocess — why "locate, don't launch"
`shutil.which("git")` searches the directories on PATH for the executable;
`subprocess.run(["git", "--version"])` actually *runs* git. For a yes/no "is it
installed?" question, `which` is the safer primitive because it's a pure filesystem
lookup — zero code executed. The moment I answer that question by running the tool,
I've executed arbitrary code just to get a boolean: it can hang waiting on input,
exit non-zero on `--version` (some tools don't support the flag → a false FAIL),
cost a process spawn, or — worst case — run a malicious binary sitting earlier on
PATH. Rule of thumb: **locate, don't launch**, when existence is all I need.

## The exit code
CI pipelines and Makefiles decide success/failure from the **exit code**, not printed
output: `exit(0)` = success, `exit(1)` = failure. If the script always exits `0`, a
broken env still reads as "passed", so downstream steps (e.g. `make setup && make train`)
kick off on a busted environment and the real failure surfaces later, somewhere confusing.

## What broke / what I got wrong first
My first two submissions were the reference `verify.py` copied, with type hints bolted
on — I mistook editing someone else's file for writing my own. It cost real time and,
worse, the copy actively lied about my machine: it checked `torch.cuda.is_available()`
(NVIDIA-only) and reported my working M4 GPU as `[FAIL]`, plus it checked libs I hadn't
installed. The lesson: on a setup task that *feels* too trivial to bother writing, the
one non-trivial bit (CUDA→MPS) is exactly the bit copying skips. Next lesson I start
from the blank file first and only open the reference to unblock a single stuck point.

## Definition of done (protocol check)
- [x] Rebuilt verify.py from a blank file, no copying
- [x] Can explain every line out loud
- [x] Passed the understanding check (3 Qs)
