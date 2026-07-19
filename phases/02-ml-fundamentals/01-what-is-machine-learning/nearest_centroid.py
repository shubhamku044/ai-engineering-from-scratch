# Phase 2 · Lesson 01 — What Is Machine Learning
# The whole idea in one classifier: instead of writing rules, LEARN them from data.
# A nearest-centroid classifier learns one prototype (mean) per class, then labels a
# new point by whichever prototype it's closest to. Then prove it beats random guessing.
# Scaffold mode: fill every `raise NotImplementedError`; the tests are the spec.
import math

# A tiny 2D dataset: class 0 clusters near (1,1), class 1 clusters near (6,6).
TRAIN_X = [[1.0, 1.0], [1.5, 0.5], [0.5, 1.5], [6.0, 6.0], [5.5, 6.5], [6.5, 5.5]]
TRAIN_Y = [0, 0, 0, 1, 1, 1]

TEST_X = [[1.2, 0.8], [0.7, 1.1], [6.2, 5.8], [5.9, 6.3]]
TEST_Y = [0, 0, 1, 1]


def fit(X: list[list[float]], y: list[int]) -> dict[int, list[float]]:
    """LEARN a prototype per class: the centroid (mean point) of each class's samples.

    Return {label: centroid}, where centroid[j] = mean of feature j over all rows
    whose label == that class. This dict IS the trained model — the "rules" learned
    from data, encoded as numbers.
    """
    # Hint: group rows by label; for each group, average each feature column.
    class_samples: dict[int, list[list[float]]] = {}
    for x, label in zip(X, y, strict=True):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(x)

    centroids: dict[int, list[float]] = {}
    for label, samples in class_samples.items():
        centroid = [
            sum(feature) / len(samples) for feature in zip(*samples, strict=True)
        ]
        centroids[label] = centroid

    return centroids


def euclidean(a: list[float], b: list[float]) -> float:
    """Straight-line distance between two points: √Σ(aᵢ − bᵢ)²."""
    return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b, strict=True)))


def predict(centroids: dict[int, list[float]], x: list[float]) -> int:
    """Classify x by the NEAREST centroid — return that centroid's label."""
    # Hint: min over centroids.items() by euclidean(x, centroid).
    closest_centroid = min(centroids.items(), key=lambda item: euclidean(x, item[1]))
    return closest_centroid[0]


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Fraction correct: (# matches) / (total)."""
    # Hint: sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return correct / total if total > 0 else 0


# ── tests (the spec — make them all pass) ────────────────────────────────────

if __name__ == "__main__":
    print("=== Training (learning centroids from data) ===")
    model = fit(TRAIN_X, TRAIN_Y)
    print("learned centroids:", model)
    # class 0's centroid should sit near (1,1), class 1's near (6,6)
    assert math.isclose(model[0][0], 1.0, abs_tol=0.5) and math.isclose(
        model[0][1], 1.0, abs_tol=0.5
    )
    assert math.isclose(model[1][0], 6.0, abs_tol=0.5) and math.isclose(
        model[1][1], 6.0, abs_tol=0.5
    )
    print("✓ centroids landed where each class clusters")

    print("\n=== Prediction ===")
    preds = [predict(model, x) for x in TEST_X]
    print("predictions:", preds, "truth:", TEST_Y)
    acc = accuracy(TEST_Y, preds)
    print(f"accuracy: {acc:.2%}")

    print("\n=== Beating the random baseline ===")
    # 2 classes → random guessing ≈ 50%. A model that learned anything must beat that.
    random_baseline = 0.5
    assert acc > random_baseline, (
        f"model ({acc}) should beat random ({random_baseline})"
    )
    assert acc == 1.0, "on this cleanly separated data, expect 100%"
    print(
        f"✓ model accuracy {acc:.0%} > random {random_baseline:.0%} — it LEARNED the pattern"
    )

    print("\nAll tests passed! 🎉")
