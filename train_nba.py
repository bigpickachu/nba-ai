"""Fase NBA-2: ML (linhas com moneyline) + Totals (janela completa). Mesmo protocolo."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
import xgboost as xgb

DB = "C:/Users/franc/nba-ai/data/nba.db"
OUT = "C:/Users/franc/nba-ai/reports"
os.makedirs(OUT, exist_ok=True)
CUTOFF = "2021-10-01"  # treino seasons 2008-2021, teste 2022-2026

WEST = {"dal", "den", "gs", "hou", "lac", "lal", "mem", "min", "no", "noh",
        "okc", "phx", "por", "sac", "sa", "sea", "utah"}
EAST = {"atl", "bkn", "bos", "cha", "chi", "cle", "det", "ind", "mia", "mil",
        "njn", "ny", "orl", "phi", "tor", "wsh"}

e = create_engine(f"sqlite:///{DB}")
df = pd.read_sql("SELECT * FROM games ORDER BY date", e)
df["date"] = pd.to_datetime(df["date"])
print("equipas:", sorted(set(df["home"]) | set(df["away"])))
unk = (set(df["home"]) | set(df["away"])) - WEST - EAST
print("sem conferencia (ficam UNK):", unk or "nenhuma")
df["conf_home"] = df["home"].apply(lambda t: "W" if t in WEST else ("E" if t in EAST else "U"))
df["conf_away"] = df["away"].apply(lambda t: "W" if t in WEST else ("E" if t in EAST else "U"))

# Features so do passado (ordens cronologica; history antes do jogo)
hist = {}
feats = []
for _, r in df.iterrows():
    d = r["date"]
    out = {}
    for side, team in [("h", r["home"]), ("a", r["away"])]:
        h = hist.get(team, [])
        last10 = h[-10:]
        out[f"{side}_win10"] = np.mean([x[1] for x in last10]) if last10 else np.nan
        out[f"{side}_diff10"] = np.mean([x[0] for x in last10]) if last10 else np.nan
        out[f"{side}_ngames"] = len(h)
        out[f"{side}_rest"] = (d - h[-1][2]).days - 1 if h else np.nan
        out[f"{side}_b2b"] = 1 if h and (d - h[-1][2]).days == 1 else 0
    feats.append(out)
    hd = r["score_home"] - r["score_away"]
    hist.setdefault(r["home"], []).append((hd, 1 if hd > 0 else 0, d))
    hist.setdefault(r["away"], []).append((-hd, 1 if hd < 0 else 0, d))
df = pd.concat([df, pd.DataFrame(feats)], axis=1)
df["conf_W"] = ((df["conf_home"] == "W") & (df["conf_away"] == "W")).astype(int)
df["conf_E"] = ((df["conf_home"] == "E") & (df["conf_away"] == "E")).astype(int)
df["playoff"] = df["playoffs"].astype(int)

BASE = ["h_win10", "h_diff10", "a_win10", "a_diff10", "h_rest", "a_rest",
        "h_b2b", "a_b2b", "playoff", "conf_W", "conf_E"]

def fit_models(Xtr, ytr, Xte):
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=4)
    clf.fit(Xtr, ytr)
    return lr.predict_proba(Xte)[:, 1], clf.predict_proba(Xte)[:, 1]

def show(nome, y, probs):
    for m, p in probs.items():
        print(f"{nome} {m}: logloss={log_loss(y, p):.4f} brier={brier_score_loss(y, p):.4f} acc={accuracy_score(y, p > 0.5):.4f}")

# ---------- MERCADO 1: MONEYLINE H/A ----------
ml = df.dropna(subset=["ml_home_dec", "ml_away_dec"]).copy()
ml["s"] = 1 / ml["ml_home_dec"] + 1 / ml["ml_away_dec"]
ml["p_mkt_h"] = (1 / ml["ml_home_dec"]) / ml["s"]
ml = ml.dropna(subset=BASE).reset_index(drop=True)
tr, te = ml[ml["date"] < CUTOFF], ml[ml["date"] >= CUTOFF]
print(f"\nML: total={len(ml)} treino={len(tr)} teste={len(te)} | teste seasons {sorted(te['season'].unique())}")
F1 = BASE + ["p_mkt_h"]
p_lr, p_xgb = fit_models(tr[F1], tr["home_win"].astype(int), te[F1])
yte = te["home_win"].astype(int).to_numpy()
print(f"ML baseline mercado (taxa casa no teste): {yte.mean():.4f}")
show("ML", yte, {"mercado": te["p_mkt_h"].to_numpy(), "logistica": p_lr, "xgboost": p_xgb})
pd.DataFrame([{"modelo": "mercado", "log_loss": log_loss(yte, te['p_mkt_h']),
               "brier": brier_score_loss(yte, te['p_mkt_h']), "acc": accuracy_score(yte, te['p_mkt_h'] > .5)},
              {"modelo": "logistica", "log_loss": log_loss(yte, p_lr),
               "brier": brier_score_loss(yte, p_lr), "acc": accuracy_score(yte, p_lr > .5)},
              {"modelo": "xgboost", "log_loss": log_loss(yte, p_xgb),
               "brier": brier_score_loss(yte, p_xgb), "acc": accuracy_score(yte, p_xgb > .5)}]
   ).to_csv(f"{OUT}/metrics_nba_ml.csv", index=False)

# ---------- MERCADO 2: TOTALS (Over; pushes void) ----------
tt = df.copy()
tt["pts"] = tt["score_home"] + tt["score_away"]
tt["push"] = tt["pts"] == tt["total"]
print(f"\nTotals pushes void: {tt['push'].sum()} ({tt['push'].mean()*100:.2f}%)")
tt = tt[~tt["push"]].copy()
tt["over"] = (tt["pts"] > tt["total"]).astype(int)
tt = tt.dropna(subset=BASE).reset_index(drop=True)
tr2, te2 = tt[tt["date"] < CUTOFF], tt[tt["date"] >= CUTOFF]
print(f"Totals: total={len(tt)} treino={len(tr2)} teste={len(te2)} | teste seasons {sorted(te2['season'].unique())}")
print(f"Totals baseline (taxa Over treino/teste): {tr2['over'].mean():.4f}/{te2['over'].mean():.4f}")
print(f"Totals mercado (MAE linha vs real, teste): {(te2['pts'] - te2['total']).abs().mean():.2f} pts")
F2 = BASE + ["total"]
p_lr2, p_xgb2 = fit_models(tr2[F2], tr2["over"], te2[F2])
y2 = te2["over"].to_numpy()
const = np.full(len(y2), tr2["over"].mean())
show("Totals", y2, {"constante": const, "logistica": p_lr2, "xgboost": p_xgb2})
pd.DataFrame([{"modelo": "constante", "log_loss": log_loss(y2, const),
               "brier": brier_score_loss(y2, const), "acc": accuracy_score(y2, const > .5)},
              {"modelo": "logistica", "log_loss": log_loss(y2, p_lr2),
               "brier": brier_score_loss(y2, p_lr2), "acc": accuracy_score(y2, p_lr2 > .5)},
              {"modelo": "xgboost", "log_loss": log_loss(y2, p_xgb2),
               "brier": brier_score_loss(y2, p_xgb2), "acc": accuracy_score(y2, p_xgb2 > .5)}]
   ).to_csv(f"{OUT}/metrics_nba_tot.csv", index=False)

# Calibracao dos vencedores
for nome, y, probs in [("ml", yte, {"mercado": te["p_mkt_h"].to_numpy(), "logistica": p_lr, "xgboost": p_xgb}),
                       ("tot", y2, {"constante": const, "logistica": p_lr2, "xgboost": p_xgb2})]:
    plt.figure()
    for m, p in probs.items():
        pt, pp = calibration_curve(y, p, n_bins=10)
        plt.plot(pp, pt, marker="o", label=m)
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.xlabel("prevista"); plt.ylabel("observada"); plt.title(f"Calibracao NBA {nome}")
    plt.legend()
    plt.savefig(f"{OUT}/calibration_nba_{nome}.png", dpi=120, bbox_inches="tight")
print(f"\nGravados em {OUT}/")
