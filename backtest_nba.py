"""Fase NBA-3: value-betting ML. A=frozen (protocolo) + B=walk-forward por eras."""
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression

DB = "C:/Users/franc/nba-ai/data/nba.db"
OUT = "C:/Users/franc/nba-ai/reports"
os.makedirs(OUT, exist_ok=True)
CUTOFF = "2021-10-01"
EV_MIN = 1.05
rng = np.random.default_rng(7)

WEST = {"dal", "den", "gs", "hou", "lac", "lal", "mem", "min", "no", "noh",
        "okc", "phx", "por", "sac", "sa", "sea", "utah"}
EAST = {"atl", "bkn", "bos", "cha", "chi", "cle", "det", "ind", "mia", "mil",
        "njn", "ny", "orl", "phi", "tor", "wsh"}

e = create_engine(f"sqlite:///{DB}")
df = pd.read_sql("SELECT * FROM games ORDER BY date", e)
df["date"] = pd.to_datetime(df["date"])
for c in ["ml_home_dec", "ml_away_dec"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["ml_home_dec", "ml_away_dec"]).copy()  # janela ML
df["s"] = 1 / df["ml_home_dec"] + 1 / df["ml_away_dec"]
df["p_mkt_h"] = (1 / df["ml_home_dec"]) / df["s"]
df["p_mkt_a"] = (1 / df["ml_away_dec"]) / df["s"]
df["conf_home"] = df["home"].apply(lambda t: "W" if t in WEST else "E")
df["conf_away"] = df["away"].apply(lambda t: "W" if t in WEST else "E")

hist = {}
recs = []
for _, r in df.iterrows():
    d = r["date"]
    o = {}
    for side, team in [("h", r["home"]), ("a", r["away"])]:
        h = hist.get(team, [])
        last10 = h[-10:]
        o[f"{side}_win10"] = np.mean([x[1] for x in last10]) if last10 else np.nan
        o[f"{side}_diff10"] = np.mean([x[0] for x in last10]) if last10 else np.nan
        o[f"{side}_rest"] = (d - h[-1][2]).days - 1 if h else np.nan
        o[f"{side}_b2b"] = 1 if h and (d - h[-1][2]).days == 1 else 0
    recs.append(o)
    hd = r["score_home"] - r["score_away"]
    hist.setdefault(r["home"], []).append((hd, 1 if hd > 0 else 0, d))
    hist.setdefault(r["away"], []).append((-hd, 1 if hd < 0 else 0, d))
df = pd.concat([df, pd.DataFrame(recs)], axis=1)
df["conf_W"] = ((df["conf_home"] == "W") & (df["conf_away"] == "W")).astype(int)
df["conf_E"] = ((df["conf_home"] == "E") & (df["conf_away"] == "E")).astype(int)
df["playoff"] = df["playoffs"].fillna(0).astype(int)
df["y"] = (df["score_home"] > df["score_away"]).astype(int)
FEAT = ["h_win10", "h_diff10", "a_win10", "a_diff10", "h_rest", "a_rest",
        "h_b2b", "a_b2b", "playoff", "conf_W", "conf_E", "p_mkt_h", "p_mkt_a"]
df = df.dropna(subset=FEAT).reset_index(drop=True)
print(f"janela ML: {len(df)} jogos, seasons {sorted(df['season'].unique())}")

def value_bets(frame, probs):
    frame = frame.copy()
    frame["p_H"], frame["p_A"] = probs[:, 0], probs[:, 1]
    frame["ev_H"], frame["ev_A"] = frame["p_H"] * frame["ml_home_dec"], frame["p_A"] * frame["ml_away_dec"]
    out = []
    for _, r in frame.iterrows():
        for side in ["H", "A"]:
            ev, p = r[f"ev_{side}"], r[f"p_{side}"]
            odd = r[f"ml_{'home' if side == 'H' else 'away'}_dec"]
            if ev > EV_MIN:
                win = (1 if side == "H" else 0) == r["y"]
                b = odd - 1
                f = (b * p - (1 - p)) / b if b > 0 else 0.0
                out.append({"date": r["date"].date().isoformat(), "season": int(r["season"]),
                    "league": "NBA", "jogo": f"{r['away']}@{r['home']}", "lado": side,
                    "odd": round(float(odd), 3), "p_model": round(float(p), 4),
                    "ev": round(float(ev), 4), "kelly_full": round(max(float(f), 0.0), 4),
                    "resultado": "H" if r["y"] == 1 else "A",
                    "lucro_1u": round(float((odd - 1) if win else -1.0), 3)})
    return pd.DataFrame(out)

def resumo(nome, profits):
    profits = np.asarray(profits, dtype=float)
    n = len(profits)
    if n == 0:
        print(f"{nome}: sem apostas"); return None
    roi = profits.sum() / n * 100
    cum = np.cumsum(profits)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    print(f"{nome}: n={n} lucro={profits.sum():+.2f}u ROI={roi:+.2f}% DDmax={dd:.2f}u")
    return {"estrategia": nome, "n": n, "lucro_u": round(float(profits.sum()), 2),
            "roi_pct": round(float(roi), 2), "dd_max_u": round(dd, 2)}

def favorite_profits(frame):
    pr = []
    for _, r in frame.iterrows():
        s = "H" if r["ml_home_dec"] < r["ml_away_dec"] else "A"
        odd = r["ml_home_dec"] if s == "H" else r["ml_away_dec"]
        win = (1 if s == "H" else 0) == r["y"]
        pr.append((odd - 1) if win else -1.0)
    return pr

# ---------- PARTE A: protocolo estrito (modelo congelado) ----------
tr = df[df["date"] < CUTOFF].copy()
te = df[df["date"] >= CUTOFF].copy().reset_index(drop=True)
print(f"\nParte A — treino {len(tr)} | teste {len(te)} (seasons {sorted(te['season'].unique())})")
lr = LogisticRegression(max_iter=1000).fit(tr[FEAT], tr["y"])
bets = value_bets(te, lr.predict_proba(te[FEAT]))
bets.to_csv(f"{OUT}/bets_nba.csv", index=False)
res = [resumo("modelo", bets["lucro_1u"]) or {"estrategia": "modelo", "n": 0}]
res.append(resumo("favorito-mercado", favorite_profits(te)))
N = len(bets)
if N > 0:
    slots = [(i, s) for i in te.index for s in ["H", "A"]]
    rrois = []
    for _ in range(500):
        idx = rng.choice(len(slots), size=N, replace=False)
        pr = []
        for k in idx:
            i, s = slots[k]
            r = te.loc[i]
            odd = r["ml_home_dec"] if s == "H" else r["ml_away_dec"]
            win = (1 if s == "H" else 0) == r["y"]
            pr.append((odd - 1) if win else -1.0)
        rrois.append(sum(pr) / N * 100)
    print(f"aleatoria: N={N} trials=500 ROI_medio={np.mean(rrois):+.2f}% "
          f"p5={np.percentile(rrois,5):+.2f}% p95={np.percentile(rrois,95):+.2f}% "
          f"P(aleat>modelo)={np.mean(np.asarray(rrois) > res[0]['roi_pct']):.3f}")
    res.append({"estrategia": "aleatoria-media500", "n": N,
        "lucro_u": round(float(np.mean(rrois) / 100 * N), 2),
        "roi_pct": round(float(np.mean(rrois)), 2), "dd_max_u": None})
print("\nModelo por season (teste):")
for s, g in bets.groupby("season"):
    print(f"  {s}: n={len(g)} ROI={g['lucro_1u'].sum()/len(g)*100:+.2f}%")

# ---------- PARTE B: walk-forward por eras (tendencia da eficiencia) ----------
print("\nParte B — walk-forward: treino=seasons anteriores, teste=season S")
eras = {"2008-2013": [], "2014-2019": [], "2020-2023": []}
fav_eras = {"2008-2013": [], "2014-2019": [], "2020-2023": []}
opp = {"2008-2013": 0, "2014-2019": 0, "2020-2023": 0}  # oportunidades por era
for S in range(2011, 2024):
    trS = df[df["season"] < S]
    teS = df[df["season"] == S]
    if len(trS) < 500 or len(teS) == 0:
        continue
    m = LogisticRegression(max_iter=1000).fit(trS[FEAT], trS["y"])
    b = value_bets(teS, m.predict_proba(teS[FEAT]))
    era = "2008-2013" if S <= 2013 else ("2014-2019" if S <= 2019 else "2020-2023")
    eras[era].extend(b["lucro_1u"].tolist())
    fav_eras[era].extend(favorite_profits(teS))
    opp[era] += len(teS)
    print(f"  season {S}: treino={len(trS)} teste={len(teS)} value_bets={len(b)}")
print("\nModelo por era:")
for era, pr in eras.items():
    a = np.asarray(pr)
    print(f"  {era}: n={len(a)} jogos-base={opp[era]} ROI={a.sum()/len(a)*100:+.2f}%")
print("Favorito-mercado por era:")
for era, pr in fav_eras.items():
    a = np.asarray(pr)
    print(f"  {era}: n={len(a)} ROI={a.sum()/len(a)*100:+.2f}%")

pd.DataFrame(res).to_csv(f"{OUT}/backtest_nba_summary.csv", index=False)
print(f"\nGravados: {OUT}/bets_nba.csv, {OUT}/backtest_nba_summary.csv")
