import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

RS, OUTER, INNER, NPERM = 42, 5, 3, 100
d = pd.read_csv("data/datasets/consolidated_dataset.csv").sort_values(["session_id","win_start_ms"])
d["win_index"] = d.groupby("session_id").cumcount()
plant = [c for c in d.columns if c.endswith("_snorm")]
assert len(plant) == 22, plant
y_col = "valence_direction"
sub = d.dropna(subset=[y_col]).copy()
y = LabelEncoder().fit_transform(sub[y_col]); g = sub["participant_id"].to_numpy(); t = sub["win_index"].to_numpy()
grid = {"randomforestclassifier__n_estimators":[200],"randomforestclassifier__max_depth":[4,None],"randomforestclassifier__min_samples_leaf":[1,3]}
def rf(): return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(class_weight="balanced", random_state=RS))

def nested(X):
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=RS); pred = np.empty_like(y)
    for tr, te in outer.split(X, y, g):
        cv = list(StratifiedGroupKFold(INNER, shuffle=True, random_state=RS).split(X[tr], y[tr], g[tr]))
        gs = GridSearchCV(rf(), grid, scoring="f1_macro", cv=cv, n_jobs=-1).fit(X[tr], y[tr]); pred[te] = gs.predict(X[te])
    return round(f1_score(y, pred, average="macro"),3), round(accuracy_score(y, pred),3)

def fixed(X, yy, est):
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=RS); pred = np.empty_like(yy)
    for tr, te in outer.split(X, yy, g):
        est.fit(X[tr], yy[tr]); pred[te] = est.predict(X[te])
    return f1_score(yy, pred, average="macro")

Xp = sub[plant].to_numpy(float); Xt = t.reshape(-1,1).astype(float); Xpt = np.hstack([Xp, Xt])
# residualise plant features on window index (pooled linear trend)
Xr = Xp.copy()
for j in range(Xp.shape[1]):
    m = ~np.isnan(Xp[:,j]); lr = LinearRegression().fit(Xt[m], Xp[m,j]); Xr[m,j] = Xp[m,j] - lr.predict(Xt[m])

out = {}
out["plant_only"] = nested(Xp); out["time_only"] = nested(Xt); out["plant_plus_time"] = nested(Xpt); out["plant_residualised"] = nested(Xr)
print(out, flush=True)

# tuned plant-only architecture, then two nulls: standard shuffle vs. shuffle within window index (time-preserving)
gs = GridSearchCV(rf(), grid, scoring="f1_macro", cv=list(StratifiedGroupKFold(OUTER, shuffle=True, random_state=RS).split(Xp, y, g)), n_jobs=-1).fit(Xp, y)
est = gs.best_estimator_; obs = fixed(Xp, y, clone(est))
rng = np.random.default_rng(RS)
null_std = np.array([fixed(Xp, rng.permutation(y), clone(est)) for _ in range(NPERM)])
def perm_within_time(yy):
    yy = yy.copy()
    for w in np.unique(t):
        idx = np.where(t == w)[0]; yy[idx] = yy[rng.permutation(idx)]
    return yy
null_time = np.array([fixed(Xp, perm_within_time(y), clone(est)) for _ in range(NPERM)])
out["perm"] = {"observed": round(obs,3), "std_null_mean": round(null_std.mean(),3), "std_p": round((np.sum(null_std>=obs)+1)/(NPERM+1),3),
               "within_time_null_mean": round(null_time.mean(),3), "within_time_null_max": round(null_time.max(),3),
               "within_time_p": round((np.sum(null_time>=obs)+1)/(NPERM+1),3), "n_exceed_within_time": int(np.sum(null_time>=obs))}
# same time-preserving test for residualised features
gs2 = GridSearchCV(rf(), grid, scoring="f1_macro", cv=list(StratifiedGroupKFold(OUTER, shuffle=True, random_state=RS).split(Xr, y, g)), n_jobs=-1).fit(Xr, y)
obs2 = fixed(Xr, y, clone(gs2.best_estimator_))
null2 = np.array([fixed(Xr, perm_within_time(y), clone(gs2.best_estimator_)) for _ in range(NPERM)])
out["perm_residualised"] = {"observed": round(obs2,3), "within_time_null_mean": round(null2.mean(),3), "within_time_p": round((np.sum(null2>=obs2)+1)/(NPERM+1),3)}
print(out)
json.dump(out, open("/tmp/claude-0/-home-claude/8c89ba71-2d65-593d-9442-eabcef9e364e/scratchpad/time_vs_plant.json","w"), indent=1, default=str)
