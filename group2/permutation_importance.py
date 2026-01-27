import numpy as np
from sklearn.metrics import accuracy_score

def permutation_importance_accuracy(model, X, y, features, n_repeats=3, n_samples=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    X_s = X.sample(min(n_samples, len(X)), random_state=random_state).copy()
    y_s = y.loc[X_s.index]

    base_pred = model.predict(X_s)
    base_acc = accuracy_score(y_s, base_pred)

    importances = {}

    for f in features:
        if f not in X_s.columns:
            continue

        drops = []
        for _ in range(n_repeats):
            X_perm = X_s.copy()
            X_perm[f] = rng.permutation(X_perm[f].values)
            pred_perm = model.predict(X_perm)
            acc_perm = accuracy_score(y_s, pred_perm)
            drops.append(base_acc - acc_perm)

        importances[f] = float(np.mean(drops))

    total = float(np.sum(list(importances.values())))
    mean_imp = float(np.mean(list(importances.values()))) if importances else 0.0

    print("[Permutation importance – accuracy drop]")
    print(f"  baseline accuracy: {base_acc:.4f}")
    print(f"  total importance over {len(importances)} features: {total:.4f}")
    print(f"  mean per-feature importance: {mean_imp:.4f}")

    return importances
