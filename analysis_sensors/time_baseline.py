import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

RS, OUTER, INNER, NPERM = 42, 5, 3, 200
d = pd.read_csv("data/datasets/consolidated_dataset.csv")
d = d.sort_values(["session_id", "win_start_ms"])
d["win_index"] = d.groupby("session_id").cumcount()
d["elapsed_s"] = (d["win_start_ms"] - d.groupby("session_id")["win_start_ms"].transform("min")) / 1000.0
FEATS = ["win_index", "elapsed_s"]

targets = {"valence": "valence_direction", "arousal": "arousal_level", "emotion": "label"}

def models():
    return {
        "RandomForest": (RandomForestClassifier(class_weight="balanced", random_state=RS),
                         {"n_estimators": [200, 500], "max_depth": [2, 4, None], "min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (XGBClassifier(random_state=RS, eval_metric="mlogloss", verbosity=0),
                    {"n_estimators": [100, 300], "max_depth": [2, 4], "learning_rate": [0.05, 0.1]}),
        "NeuralNet": (make_pipeline(StandardScaler(), MLPClassifier(max_iter=2000, random_state=RS)),
                      {"mlpclassifier__hidden_layer_sizes": [(16,), (32,), (32, 16)], "mlpclassifier__alpha": [1e-4, 1e-2]}),
    }

def nested(X, y, groups, est, grid, seed=RS):
    outer = StratifiedGroupKFold(n_splits=OUTER, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for tr, te in outer.split(X, y, groups):
        inner = StratifiedGroupKFold(n_splits=INNER, shuffle=True, random_state=seed)
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=list(inner.split(X[tr], y[tr], groups[tr])), n_jobs=-1)
        gs.fit(X[tr], y[tr])
        pred[te] = gs.predict(X[te])
    return f1_score(y, pred, average="macro"), accuracy_score(y, pred)

def fixed_cv(X, y, groups, est):
    outer = StratifiedGroupKFold(n_splits=OUTER, shuffle=True, random_state=RS)
    pred = np.empty_like(y)
    for tr, te in outer.split(X, y, groups):
        est.fit(X[tr], y[tr]); pred[te] = est.predict(X[te])
    return f1_score(y, pred, average="macro")

out = {}
for tname, col in targets.items():
    sub = d.dropna(subset=[col])
    X = sub[FEATS].to_numpy(float); y = LabelEncoder().fit_transform(sub[col]); g = sub["participant_id"].to_numpy()
    res = {}
    for mname, (est, grid) in models().items():
        f1, acc = nested(X, y, g, est, grid)
        res[mname] = {"macro_f1": round(f1, 3), "accuracy": round(acc, 3)}
        print(tname, mname, res[mname], flush=True)
    best = max(res, key=lambda m: res[m]["macro_f1"])
    # tune winner on full data once (as in students' pipeline), then permutation test with architecture fixed
    est, grid = models()[best]
    gs = GridSearchCV(est, grid, scoring="f1_macro",
                      cv=list(StratifiedGroupKFold(n_splits=OUTER, shuffle=True, random_state=RS).split(X, y, g)), n_jobs=-1).fit(X, y)
    from sklearn.base import clone
    obs = fixed_cv(X, y, g, clone(gs.best_estimator_))
    rng = np.random.default_rng(RS)
    null = np.array([fixed_cv(X, rng.permutation(y), g, clone(gs.best_estimator_)) for _ in range(NPERM)])
    p = (np.sum(null >= obs) + 1) / (NPERM + 1)
    k = len(np.unique(y))
    out[tname] = {"n": int(len(y)), "classes": k, "chance_macro_f1": round(1 / k, 3), "models": res, "best": best,
                  "best_params": gs.best_params_, "observed_fixed": round(obs, 3), "perm_mean": round(null.mean(), 3),
                  "perm_max": round(null.max(), 3), "n_exceed": int(np.sum(null >= obs)), "p": round(p, 3)}
    print(tname, out[tname], flush=True)
json.dump(out, open("/tmp/claude-0/-home-claude/8c89ba71-2d65-593d-9442-eabcef9e364e/scratchpad/time_baseline.json", "w"), indent=1, default=str)
