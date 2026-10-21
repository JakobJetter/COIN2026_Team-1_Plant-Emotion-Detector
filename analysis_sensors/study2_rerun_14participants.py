"""Study 2 re-analysis on 14 participants (proband_09 excluded: no retrospective consent).
Replicates the students' pipeline (nested participant-grouped StratifiedGroupKFold 5/3, macro-F1, 200 permutations
with fixed architecture) for RF/XGB/MLP x raw/snorm on the three targets, plus the time-only baseline,
the time-preserving permutation test, and the CO2 partial-correlation analysis."""
import numpy as np, pandas as pd, json, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.base import clone
from scipy.stats import spearmanr
from xgboost import XGBClassifier

RS, OUTER, INNER, NPERM = 42, 5, 3, 200
ONLY = sys.argv[1:] or None
d = pd.read_csv("data/datasets/consolidated_dataset.csv").sort_values(["session_id", "win_start_ms"])
d = d[d["participant_id"] != "proband_09"].reset_index(drop=True)
d["win_index"] = d.groupby("session_id").cumcount()
d["elapsed_s"] = (d["win_start_ms"] - d.groupby("session_id")["win_start_ms"].transform("min")) / 1000
snorm = [c for c in d.columns if c.endswith("_snorm")]
raw = [c[:-6] for c in snorm]
assert len(snorm) == 22 and all(c in d.columns for c in raw)
targets = {"valence": "valence_direction", "arousal": "arousal_level", "emotion": "label"}
print("windows", len(d), "participants", d["participant_id"].nunique())
for t, c in targets.items(): print(t, d[c].value_counts().to_dict())

def models():
    imp = SimpleImputer(strategy="median")
    return {
        "RandomForest": (make_pipeline(clone(imp), RandomForestClassifier(class_weight="balanced", random_state=RS)),
                         {"randomforestclassifier__n_estimators": [200, 500], "randomforestclassifier__max_depth": [4, 8, None], "randomforestclassifier__min_samples_leaf": [1, 3]}),
        "XGBoost": (make_pipeline(clone(imp), XGBClassifier(random_state=RS, eval_metric="mlogloss", verbosity=0)),
                    {"xgbclassifier__n_estimators": [100, 300], "xgbclassifier__max_depth": [2, 4], "xgbclassifier__learning_rate": [0.05, 0.1]}),
        "NeuralNet": (make_pipeline(clone(imp), StandardScaler(), MLPClassifier(max_iter=2000, random_state=RS)),
                      {"mlpclassifier__hidden_layer_sizes": [(32,), (64, 32)], "mlpclassifier__alpha": [1e-4, 1e-2]}),
    }

def outer_cv(y, g): return StratifiedGroupKFold(OUTER, shuffle=True, random_state=RS).split(np.zeros(len(y)), y, g)
def nested(X, y, g, est, grid):
    pred = np.empty_like(y)
    for tr, te in outer_cv(y, g):
        cv = list(StratifiedGroupKFold(INNER, shuffle=True, random_state=RS).split(X[tr], y[tr], g[tr]))
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=-1).fit(X[tr], y[tr]); pred[te] = gs.predict(X[te])
    return pred
def fixed(X, y, g, est):
    pred = np.empty_like(y)
    for tr, te in outer_cv(y, g): est.fit(X[tr], y[tr]); pred[te] = est.predict(X[te])
    return f1_score(y, pred, average="macro")

out = {}
for tname, col in targets.items():
    if ONLY and tname not in ONLY: continue
    sub = d.dropna(subset=[col]); le = LabelEncoder(); y = le.fit_transform(sub[col]); g = sub["participant_id"].to_numpy()
    res = {}
    for rep, cols in (("raw", raw), ("snorm", snorm)):
        X = sub[cols].to_numpy(float)
        for mname, (est, grid) in models().items():
            pred = nested(X, y, g, est, grid)
            res[f"{mname}/{rep}"] = {"macro_f1": round(f1_score(y, pred, average="macro"), 3), "accuracy": round(accuracy_score(y, pred), 3),
                                     "per_class_f1": {k: round(v["f1-score"], 2) for k, v in classification_report(y, pred, target_names=le.classes_, output_dict=True).items() if k in le.classes_}}
            print(tname, mname, rep, res[f"{mname}/{rep}"]["macro_f1"], flush=True)
    best = max(res, key=lambda k: res[k]["macro_f1"]); mname, rep = best.split("/")
    X = sub[raw if rep == "raw" else snorm].to_numpy(float); est, grid = models()[mname]
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=list(outer_cv(y, g)), n_jobs=-1).fit(X, y); arch = gs.best_estimator_
    obs = fixed(X, y, g, clone(arch)); rng = np.random.default_rng(RS)
    null = np.array([fixed(X, rng.permutation(y), g, clone(arch)) for _ in range(NPERM)])
    # time-only baseline (best of three, nested) + permutation
    Xt = sub[["win_index", "elapsed_s"]].to_numpy(float); tres = {}
    for mn, (e, gr) in models().items():
        p = nested(Xt, y, g, e, gr); tres[mn] = (round(f1_score(y, p, average="macro"), 3), round(accuracy_score(y, p), 3))
    tb = max(tres, key=lambda k: tres[k][0]); e, gr = models()[tb]
    tarch = GridSearchCV(e, gr, scoring="f1_macro", cv=list(outer_cv(y, g)), n_jobs=-1).fit(Xt, y).best_estimator_
    tobs = fixed(Xt, y, g, clone(tarch)); tnull = np.array([fixed(Xt, rng.permutation(y), g, clone(tarch)) for _ in range(NPERM)])
    entry = {"n": int(len(y)), "best": best, "nested": res[best], "fixed_obs": round(obs, 3), "perm_mean": round(null.mean(), 3), "perm_max": round(null.max(), 3),
             "n_exceed": int((null >= obs).sum()), "p": round(((null >= obs).sum() + 1) / (NPERM + 1), 3), "all": res,
             "time_only": {"models": tres, "best": tb, "fixed_obs": round(tobs, 3), "p": round(((tnull >= tobs).sum() + 1) / (NPERM + 1), 3)}}
    if tname == "valence":
        t = sub["win_index"].to_numpy()
        def perm_time(yy):
            yy = yy.copy()
            for w in np.unique(t):
                idx = np.where(t == w)[0]; yy[idx] = yy[rng.permutation(idx)]
            return yy
        tnull2 = np.array([fixed(X, perm_time(y), g, clone(arch)) for _ in range(NPERM)])
        entry["time_preserving"] = {"obs": round(obs, 3), "null_mean": round(tnull2.mean(), 3), "null_max": round(tnull2.max(), 3),
                                    "n_exceed": int((tnull2 >= obs).sum()), "p": round(((tnull2 >= obs).sum() + 1) / (NPERM + 1), 3)}
        Xpt = np.hstack([X, Xt]); pj = nested(Xpt, y, g, *models()[mname]); entry["plant_plus_time_f1"] = round(f1_score(y, pj, average="macro"), 3)
        # CO2 partial correlation (raw and session-normalised)
        v = sub["fer_valence"].to_numpy(float); co2 = sub["co2"].to_numpy(float)
        def pcorr(x, yv, z):
            m = ~(np.isnan(x) | np.isnan(yv) | np.isnan(z)); x, yv, z = x[m], yv[m], z[m]
            rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = yv - np.polyval(np.polyfit(z, yv, 1), z)
            return spearmanr(rx, ry)
        for lab, cols, vv, cc in (("raw", raw, v, co2), ("snorm", snorm, v - sub.groupby("session_id")["fer_valence"].transform("median").to_numpy(), co2 - sub.groupby("session_id")["co2"].transform("median").to_numpy())):
            pv, pc, pvz = [], [], []
            for c in cols:
                x = sub[c].to_numpy(float); m = ~np.isnan(x)
                pv.append(spearmanr(x[m], vv[m])); pc.append(spearmanr(x[m], cc[m])); pvz.append(pcorr(x, vv, cc))
            cv_ = spearmanr(cc[~np.isnan(cc)], vv[~np.isnan(cc)])
            entry[f"co2_{lab}"] = {"plant_val_mean_abs_rho": round(np.mean([abs(r.statistic) for r in pv]), 3), "plant_val_sig": int(sum(r.pvalue < .05 for r in pv)),
                                   "plant_co2_mean_abs_rho": round(np.mean([abs(r.statistic) for r in pc]), 3), "plant_co2_sig": int(sum(r.pvalue < .05 for r in pc)),
                                   "co2_val_rho": round(cv_.statistic, 3), "co2_val_p": float(f"{cv_.pvalue:.1g}"),
                                   "partial_mean_abs_rho": round(np.mean([abs(r.statistic) for r in pvz]), 3), "partial_sig": int(sum(r.pvalue < .05 for r in pvz))}
    out[tname] = entry; print(tname, "done", flush=True)
    json.dump(out, open(f"res14_{tname}.json", "w"), indent=1, default=str)
