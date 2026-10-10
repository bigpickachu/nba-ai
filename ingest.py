"""Ingestao NBA: CSV Kaggle -> SQLite, idempotente por game_id. So american->decimal."""
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, Date
from sqlalchemy.orm import declarative_base

DB = "C:/Users/franc/nba-ai/data/nba.db"
engine = create_engine(f"sqlite:///{DB}")
Base = declarative_base()

class Game(Base):
    __tablename__ = "games"
    game_id = Column(String, primary_key=True)  # {aaaammdd}{casa}{fora}
    date = Column(Date)
    season = Column(Integer)
    playoffs = Column(Boolean)
    home = Column(String)
    away = Column(String)
    score_home = Column(Integer)
    score_away = Column(Integer)
    home_win = Column(Boolean)
    favored = Column(String)  # home | away
    spread = Column(Float)
    total = Column(Float)
    ml_home_us = Column(Float)  # americana original
    ml_away_us = Column(Float)
    ml_home_dec = Column(Float)  # decimal convertida
    ml_away_dec = Column(Float)

def us_to_dec(o):
    if pd.isna(o):
        return None
    o = float(o)
    return 1 + 100 / abs(o) if o < 0 else 1 + o / 100

Base.metadata.create_all(engine)
df = pd.read_csv("C:/Users/franc/nba-ai/data/nba_betting.csv")
df["date"] = pd.to_datetime(df["date"], format="ISO8601")
df["game_id"] = df["date"].dt.strftime("%Y%m%d") + df["home"] + df["away"]
df["home_win"] = df["score_home"] > df["score_away"]
df["favored"] = df["whos_favored"].map({"home": "home", "away": "away"})
df["ml_home_dec"] = df["moneyline_home"].apply(us_to_dec)
df["ml_away_dec"] = df["moneyline_away"].apply(us_to_dec)

out = df.rename(columns={"moneyline_home": "ml_home_us", "moneyline_away": "ml_away_us"})[
    ["game_id", "date", "season", "playoffs", "home", "away", "score_home",
     "score_away", "home_win", "favored", "spread", "total",
     "ml_home_us", "ml_away_us", "ml_home_dec", "ml_away_dec"]]
out.to_sql("games", engine, if_exists="append", index=False)
print(f"linhas: {len(out)} | nulos ml_dec: {out['ml_home_dec'].isna().sum()}")

# Exemplos reais da conversao
print("\nvalidacao us->decimal:")
for o in [-1400.0, -120.0, 100.0, 900.0]:
    print(f"  {o:+.0f} americana -> {us_to_dec(o):.4f} decimal")
chk = out.dropna(subset=["ml_home_us"]).head(2)
for _, r in chk.iterrows():
    print(f"  {r['away']}@{r['home']}: ml_us {r['ml_away_us']:+.0f}/{r['ml_home_us']:+.0f} -> dec {r['ml_away_dec']:.4f}/{r['ml_home_dec']:.4f}")
