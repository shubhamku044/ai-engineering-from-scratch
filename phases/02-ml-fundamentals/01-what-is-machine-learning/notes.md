# Phase 2 · Lesson 01 — What Is Machine Learning · notes

## The core idea
Traditional programming: I write the rules. ML: I give data + expected outputs, and the
algorithm *learns* the rules. The trained model **is** the rules, stored as numbers. My
`fit()` returning per-class centroids is exactly this — the centroid dict is the learned rule.

## The three types
- **Supervised** — labeled data, learn input → output (spam/not-spam). *Nearest-centroid is this.*
- **Unsupervised** — no labels, find structure (clustering, PCA).
- **Reinforcement** — agent acts, gets rewards, learns a policy by trial and error.

## Classification vs regression (both supervised)
- **Classification** → predict a category. Loss: **cross-entropy** (built in Phase 1 · 06).
- **Regression** → predict a number. Loss: **MSE** (built in Phase 1 · 04).

## When NOT to use ML
If rules are few, stable, and known → just write the rules (cheaper, exact, debuggable).
ML earns its place when rules are too many, keep changing, or can't be articulated
(recognizing cats, ranking results).

## Nearest centroid (what I built)
Learn the mean point of each class; classify a new point by the nearest class mean.
Simplest "learn from data" there is — and a real baseline to beat with fancier models.

## Further watching (verified links)
Enough for THIS lesson — one 12-min video (or none, if the concepts above already landed):
- **StatQuest — A Gentle Introduction to Machine Learning** (covers all of lesson 01):
  https://www.youtube.com/watch?v=Gv9_4yMHFhI

For LATER in Phase 2 (not needed now — a reference to return to lesson by lesson):
- **StatQuest — Machine Learning playlist**:
  https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
- 3 types of ML: https://www.youtube.com/watch?v=1FZ0A1QCMWc ·
  regression vs classification: https://www.youtube.com/watch?v=E3jaFfP5BAg

## Understanding check (answered)
- **Supervised:** predict used-car price from mileage/age/model (data has the true price).
- **Unsupervised:** group music listeners by listening habits, no labels — algorithm finds groups.
- **Reinforcement:** warehouse robot learning fastest routes via rewards (fast delivery) / penalties (collisions).
- **Classification vs regression:** category → cross-entropy; continuous number → MSE.
- **Rules beat ML:** password validation (≥8 chars + a number) — rules are exact, cheap,
  maintainable; no data/training needed. ML is for when rules are too complex or can't be written.
