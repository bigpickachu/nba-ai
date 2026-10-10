import pandas as pd
df = pd.read_csv("C:/Users/franc/nba-ai/data/nba_betting.csv")
df["date"] = pd.to_datetime(df["date"])
print("range:", df["date"].min().date(), "->", df["date"].max().date())
print("\njogos por season:")
print(df["season"].value_counts().sort_index())
print("\nmoneyline nula por season (%):")
print((df["moneyline_home"].isna().groupby(df["season"]).mean() * 100).round(1))
print("\nplayoffs vs regular:", df["playoffs"].sum(), "playoff /", (~df["playoffs"]).sum(), "regular")
print("\nduplicados (date+home+away):", df.duplicated(["date", "home", "away"]).sum())
print("\nhome win %:", (df["score_home"] > df["score_away"]).mean().round(4))
print("home win % regular:", (df.loc[~df["playoffs"], "score_home"] > df.loc[~df["playoffs"], "score_away"]).mean().round(4))
# Back-to-backs: equipa jogou no dia anterior?
played = set()
rows = []
for d, g in df.sort_values("date").groupby("date"):
    teams = set(g["home"]) | set(g["away"])
    b2b_home = g["home"].isin(played & {t for t in teams}).sum()  # simplificado abaixo
    rows.append((d, teams))
    played |= teams
# calculo correto: para cada jogo, home/away estavam no set do dia anterior
prev = set()
n = hb = ab = either = 0
for d, g in df.sort_values("date").groupby("date"):
    teams = set(g["home"]) | set(g["away"])
    for _, r in g.iterrows():
        n += 1
        h = r["home"] in prev
        a = r["away"] in prev
        hb += h
        ab += a
        either += (h or a)
    prev = teams
print(f"\njogos: {n} | casa em B2B: {hb} ({hb/n*100:.1f}%) | fora em B2B: {ab} ({ab/n*100:.1f}%) | algum em B2B: {either} ({either/n*100:.1f}%)")
