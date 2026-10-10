# NBA: o mesmo protocolo, refutado por outra via (comparativo futebol vs NBA)

Replica do protocolo de `football-ai` na NBA (24.440 jogos, 2007–2026): ingestão, EDA, modelos, value-betting. Resultado: refutação por **irrelevância** — o modelo empata o ruído em vez de perder para ele.

## NBA-1 — Dados (`ingest.py`, `inspect_csv.py`, `analyze_csv.py`)

Dataset Kaggle "NBA Betting Data" (spreads + totals + moneylines americanas, closing lines). SQLite `data/nba.db`, PK `{aaaammdd}{casa}{fora}`, idempotente. Conversão americana→decimal validada (-1400→1.0714, +100→2.0000).
Validações: efeito casa **58.0%** (vs ~45% no futebol); back-to-backs **35.7%** (casa 16.8%, fora 27.1% — a fadiga penaliza o visitante); moneylines **ausentes em 2024–2026** → protocolo dividido: ML nas seasons com cobertura (n=19.791), totals na janela completa (n=24.142).

## NBA-2 — Modelos (`train_nba.py`)

Split <2021-10-01 / >=2021-10-01, features pré-jogo (forma, descanso/B2B, conferência + mercado), sem box scores (sem 4 fatores — limitação registada).
ML: mercado 0.6177 — logística 0.6187 — XGBoost 0.6194. Totals: constante 0.6933 = logística; XGBoost pior. Com 17.804 jogos de treino o argumento "pouca amostra" morre: era o mercado, não os dados.

## NBA-3 — Value-betting (`backtest_nba.py`)

EV>1.05, flat 1u, teste ML: modelo **ROI -0.14%** (n=1.881) vs favorito -3.06% vs aleatória -2.34% (P(aleat>modelo)=0.16). Empate técnico com o ruído: o modelo aposta em 95% dos jogos, 100% underdogs — o threshold seleciona cauda, não valor. Walk-forward por eras: -6.20% → -2.39% → +1.24% (SE ±2.3%: zero); favorito estável (-3.6%/-4.6%/-4.3%): eficiência estacionária, sem deriva temporal.

## O comparativo que fecha o multi-desporto

| | Futebol | NBA |
|---|---|---|
| Gap modelo–mercado | +0.0025 log-loss | +0.0010 — ainda mais fino |
| Backtest | -15.84%, seletivo (478) | -0.14%, tudo underdog (1.881) |
| Vs aleatória | perde (P=0.82) | empata (P=0.16) |
| Veredicto | refutado com prejuízo | refutado por irrelevância |

Os 225 pontos diluem o ruído (accuracy do mercado 66.8% vs 56%) mas o mercado precifica-o na totalidade. Sem box scores/escalações/xG pré-jogo, não há edge em nenhum dos dois desportos.

## Reproduzir

```bash
pip install pandas sqlalchemy requests scikit-learn xgboost matplotlib
# pôr nba_betting.csv (Kaggle) em data/ e:
python ingest.py && python train_nba.py && python backtest_nba.py
```

`data/` e `*.db` fora do repo (ver `.gitignore`).
