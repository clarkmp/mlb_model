"""
main.py
-------
MLB Betting Model — main entry point.

Usage:
    python main.py                        # full train + today's picks
    python main.py --mode train           # retrain only
    python main.py --mode live            # today's picks only (needs saved model)
    python main.py --bankroll 5000        # set bankroll
    python main.py --min-edge 0.05        # require 5% edge
    python main.py --seasons 2022 2023 2024
"""

import os
import argparse
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from mlb_data import (
    fetch_season_schedule, fetch_team_season_stats,
    fetch_today_schedule, fetch_mlb_odds,
    fetch_pitcher_stats, fetch_lineup_stats,
)
from mlb_features import build_features, build_game_features, TARGET_COL
import mlb_features as _feat_module
from mlb_model import (
    walk_forward_backtest, evaluate_model, train_final_model,
    save_model, load_model, predict_proba, get_feature_importance,
)
from mlb_betting import (
    evaluate_game, run_backtest,
    edge_buckets, monthly_performance,
)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# _feature_cols() is set dynamically by build_features() — always read from module
def _feature_cols():
    return _feat_module.FEATURE_COLS

CONFIG = {
    "seasons":        [2023, 2024],
    "initial_bankroll": 1000.0,
    "min_edge":        0.04,
    "kelly_frac":      0.25,
    "max_stake_pct":   0.04,
    "min_train_games": 300,
    "step_games":       30,
}

W = 110   # total output width


# ─────────────────────────────────────────────
# Print helpers — clean, aligned output
# ─────────────────────────────────────────────

def header(title: str):
    print(f"\n{'─'*W}")
    print(f"  {title.upper()}")
    print(f"{'─'*W}")

def kv(label: str, value, width: int = 32):
    print(f"  {label:<{width}} {value}")

def divider():
    print(f"  {'─'*(W-4)}")


# ─────────────────────────────────────────────
# Data loading with caching
# ─────────────────────────────────────────────

def load_games(seasons: list[int]) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"mlb_games_{'_'.join(map(str,seasons))}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["game_date"])
        print(f"  Loaded {len(df)} games from cache. Delete {cache_file} to refresh.")
        return df
    print("  Fetching game history from MLB Stats API...")
    df = fetch_season_schedule(seasons)
    if not df.empty:
        df.to_csv(cache_file, index=False)
    return df


def load_team_stats(seasons: list[int]) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"mlb_team_stats_{'_'.join(map(str,seasons))}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        print(f"  Loaded team stats from cache ({len(df)} rows).")
        return df
    dfs = []
    for s in seasons:
        try:
            dfs.append(fetch_team_season_stats(s))
            time.sleep(0.3)
        except Exception as e:
            print(f"  Warning: could not fetch {s} team stats: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(cache_file, index=False)
    return df


def load_odds_for_history(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach approximate odds to historical games for backtest.
    Synthesises moneyline odds from each team's rolling win rate and
    park factor — no merge, no groupby, just vectorised operations.
    """
    if "home_odds" in games_df.columns and games_df["home_odds"].notna().sum() > 100:
        return games_df

    print("  Synthesising historical odds from team win rates + park factors...")
    from mlb_data import PARK_FACTORS

    df = games_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Rolling 60-game home win rate per team, computed without lookahead
    df = df.sort_values("game_date").reset_index(drop=True)
    df["_roll_wr"] = (
        df.groupby("home_team")["home_win"]
        .transform(lambda x: x.shift(1).rolling(60, min_periods=10).mean())
        .fillna(0.54)   # league-average home win rate as cold-start
    )

    # Park factor nudge (already a ratio, e.g. 1.08 for Fenway)
    pf = df["home_team"].map(PARK_FACTORS).fillna(1.00)

    # Blend rolling win rate with league average + park factor
    raw_prob = ((df["_roll_wr"] * 0.65 + 0.54 * 0.35) * pf).clip(0.30, 0.72)

    # Small random noise so every game doesn't get identical odds
    rng = np.random.default_rng(42)
    noisy_prob = (raw_prob + rng.normal(0, 0.02, len(df))).clip(0.28, 0.74)

    def to_american(p: float) -> int:
        p_vig = float(p) * 0.952   # ~5% vig
        p_vig = max(0.10, min(0.90, p_vig))
        if p_vig >= 0.5:
            return -int(round(p_vig / (1 - p_vig) * 100 / 5) * 5)
        else:
            return  int(round((1 - p_vig) / p_vig * 100 / 5) * 5)

    df["home_odds"] = noisy_prob.apply(to_american)
    df["away_odds"] = (1 - noisy_prob).apply(to_american)
    df = df.drop(columns=["_roll_wr"])
    return df


# ─────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────

def run_train(seasons: list[int]) -> tuple:
    header("1 — loading game history")
    games_raw = load_games(seasons)
    if games_raw.empty:
        print("  ERROR: No game data found.")
        return None, None, None

    games_raw = load_odds_for_history(games_raw)
    kv("Games loaded:", len(games_raw))
    kv("Seasons:", ", ".join(str(s) for s in games_raw["game_date"].dt.year.unique()))
    kv("Teams:", games_raw["home_team"].nunique())
    kv("Home win rate:", f"{games_raw['home_win'].mean():.1%}")

    header("2 — loading team stats")
    team_stats = load_team_stats(seasons)

    header("3 — building features")
    df = build_features(games_raw, team_stats)
    n_valid = df.dropna(subset=_feature_cols()).shape[0]
    kv("Complete feature rows:", n_valid)
    kv("Features:", len(_feature_cols()))

    if n_valid < CONFIG["min_train_games"] + 50:
        print(f"\n  WARNING: Only {n_valid} complete rows — need {CONFIG['min_train_games']+50}+")
        print("  Try adding more seasons: --seasons 2022 2023 2024")

    header("4 — walk-forward model training")
    df = walk_forward_backtest(
        df, _feature_cols(), TARGET_COL,
        min_train_games = CONFIG["min_train_games"],
        step_games      = CONFIG["step_games"],
    )

    header("5 — model quality metrics")
    if df["model_prob"].notna().sum() == 0:
        print("  No predictions generated. Check that min_train_games is below dataset size.")
        return None, None, None
    metrics = evaluate_model(df)
    kv("Games evaluated:",    metrics["n_predictions"])
    kv("Home win rate:",       f"{metrics['home_win_rate']:.1%}")
    kv("ROC-AUC:",             metrics["roc_auc"],         )
    kv("Brier skill score:",   metrics["brier_skill_score"])
    kv("Log loss:",            metrics["log_loss"])
    kv("Forecast resolution:", metrics["forecast_resolution"])
    print()
    print("  Calibration (predicted → actual win rate):")
    for pred, actual in metrics["calibration_curve"]:
        bar_p = "░" * int(pred * 20)
        bar_a = "█" * int(actual * 20)
        diff = actual - pred
        sign = "+" if diff >= 0 else ""
        print(f"    {pred:.2f}  {bar_p:<20}  actual={actual:.2f}  {bar_a:<20}  {sign}{diff:.3f}")

    if metrics["roc_auc"] > 0.55:
        print(f"\n  ✓ AUC {metrics['roc_auc']} — model has meaningful predictive power")
    if metrics["brier_skill_score"] > 0:
        print(f"  ✓ Brier skill {metrics['brier_skill_score']} — beats naive baseline")

    header("6 — betting backtest")
    ledger, summary = run_backtest(
        df,
        initial_bankroll = CONFIG["initial_bankroll"],
        min_edge         = CONFIG["min_edge"],
        kelly_frac       = CONFIG["kelly_frac"],
        max_stake_pct    = CONFIG["max_stake_pct"],
    )

    if "error" in summary:
        print(f"  {summary['error']}")
    else:
        # Main metrics
        divider()
        kv("Total bets:",           summary["total_bets"])
        kv("Win / Loss:",           f"{summary['wins']} W  {summary['losses']} L")
        kv("Win rate:",             f"{summary['win_rate']:.1%}")
        kv("Avg edge:",             f"{summary['avg_edge_pct']}%")
        kv("Avg odds:",             f"+{summary['avg_odds']:.0f}" if summary['avg_odds'] >= 0 else f"{summary['avg_odds']:.0f}")
        divider()
        kv("Initial bankroll:",     f"${CONFIG['initial_bankroll']:,.2f}")
        kv("Final bankroll (Kelly):", f"${summary['final_bankroll_kelly']:,.2f}")
        kv("Total P&L (Kelly):",    f"${summary['total_pnl_kelly']:,.2f}")
        kv("ROI % (Kelly):",        f"{summary['roi_pct_kelly']}%")
        kv("Max drawdown:",         f"{summary['max_drawdown_pct']}%")
        kv("Profit factor:",        summary["profit_factor"])
        divider()
        kv("Flat stake ROI %:",     f"{summary['flat_roi_pct']}%")
        kv("Flat stake final:",     f"${summary['flat_final_bankroll']:,.2f}")
        divider()
        kv("Max win streak:",       summary["max_win_streak"])
        kv("Max loss streak:",      summary["max_loss_streak"])

        # Save summary for live mode to use
        with open(CACHE_DIR / "last_summary.json", "w") as f:
            json.dump({k: v for k, v in summary.items()
                       if not isinstance(v, (list, dict))}, f)

        header("6a — edge bucket analysis")
        buckets = edge_buckets(ledger)
        if not buckets.empty:
            print(buckets.to_string())

        header("6b — monthly performance")
        monthly = monthly_performance(ledger)
        if not monthly.empty:
            print(monthly.to_string())

    header("7 — feature importance")
    final_model = train_final_model(df, _feature_cols(), TARGET_COL)
    imp = get_feature_importance(final_model, _feature_cols())
    if not imp.empty:
        print("  Top 15 features:")
        for feat, score in imp.head(15).items():
            bar = "█" * int(score * 300)
            print(f"    {feat:<45} {score:.4f}  {bar}")

    save_model(final_model)
    return final_model, df, games_raw


# ─────────────────────────────────────────────
# Live scoring
# ─────────────────────────────────────────────

def run_live(model, history_features_df: pd.DataFrame):
    header("today's mlb games — live picks")

    # Load bankroll from last backtest
    bankroll = CONFIG["initial_bankroll"]
    summary_path = CACHE_DIR / "last_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        bankroll = s.get("final_bankroll_kelly", bankroll)

    # Fetch today's schedule
    today_df = fetch_today_schedule()
    if today_df.empty:
        print("  No MLB games scheduled today.")
        return

    # Fetch live odds
    odds_df = fetch_mlb_odds()

    # Match odds to games by team name
    def match_odds(home, away):
        if odds_df.empty:
            return None, None, 0
        # Try exact match first, then partial
        mask = (odds_df["home_team"] == home) & (odds_df["away_team"] == away)
        if not mask.any():
            mask = odds_df["home_team"].str.contains(home.split()[-1], case=False, na=False) & \
                   odds_df["away_team"].str.contains(away.split()[-1], case=False, na=False)
        row = odds_df[mask]
        if row.empty:
            return None, None, 0
        r = row.iloc[0]
        return r["home_odds_best"], r["away_odds_best"], int(r["n_books"])

    # Print table header
    print()
    col_w = W - 4
    print(f"  {'MATCHUP':<46} {'SP (HOME)':<22} {'SP (AWAY)':<22} {'ODDS':>6}  {'MODEL':>6}  {'FAIR':>6}  {'EDGE':>6}  {'STAKE':>9}  VERDICT")
    print(f"  {'─'*col_w}")

    recs = []
    for _, game in today_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        home_sp_name = str(game.get("home_sp_name", "TBD"))
        away_sp_name = str(game.get("away_sp_name", "TBD"))

        # Fetch pitcher stats (current season)
        season = pd.Timestamp.now().year
        home_sp_stats = {}
        away_sp_stats = {}
        try:
            if game.get("home_sp_id"):
                home_sp_stats = fetch_pitcher_stats(int(game["home_sp_id"]), season)
        except Exception:
            pass
        try:
            if game.get("away_sp_id"):
                away_sp_stats = fetch_pitcher_stats(int(game["away_sp_id"]), season)
        except Exception:
            pass

        # Build feature vector
        feat = build_game_features(
            home_team       = home,
            away_team       = away,
            history_df      = history_features_df,
            home_sp_stats   = home_sp_stats or None,
            away_sp_stats   = away_sp_stats or None,
        )

        model_prob = predict_proba(model, feat, _feature_cols())

        # Get odds
        home_odds, away_odds, n_books = match_odds(home, away)

        if home_odds is None:
            # No odds available — show model prob only
            matchup = f"{away} @ {home}"
            print(f"  {matchup:<46} {home_sp_name:<22} {away_sp_name:<22}  "
                  f"no odds  Model:{model_prob*100:.1f}%  (no odds available)")
            continue

        rec = evaluate_game(
            game_pk         = int(game.get("game_pk", 0)),
            home_team       = home,
            away_team       = away,
            home_sp         = home_sp_name,
            away_sp         = away_sp_name,
            model_prob_home = model_prob,
            home_odds       = home_odds,
            away_odds       = away_odds,
            bankroll        = bankroll,
            n_books         = n_books,
            min_edge        = CONFIG["min_edge"],
            kelly_frac      = CONFIG["kelly_frac"],
            max_stake_pct   = CONFIG["max_stake_pct"],
        )
        recs.append(rec)

        # Format output row
        matchup    = f"{away} @ {home}"
        odds_str   = f"{int(rec.american_odds):>+d}"
        edge_str   = f"{'+'if rec.edge>=0 else''}{rec.edge*100:.1f}%"
        stake_str  = f"${rec.stake:>8.2f}" if rec.verdict in ("BET","LEAN") else f"{'—':>9}"
        model_str  = f"{model_prob*100:.1f}%"
        fair_str   = f"{rec.fair_prob*100:.1f}%"
        verdict_tag = {
            "BET":      "★ BET ★",
            "LEAN":     "~ lean ~",
            "SKIP":     "skip",
            "NO_VALUE": "no edge",
        }.get(rec.verdict, rec.verdict)

        print(f"  {matchup:<46} {home_sp_name:<22} {away_sp_name:<22} "
              f"{odds_str:>6}  {model_str:>6}  {fair_str:>6}  {edge_str:>6}  {stake_str}  {verdict_tag}")

        if rec.notes:
            print(f"  {'':46} {'':22} {'':22}  Note: {' | '.join(rec.notes)}")

    # Summary footer
    print(f"  {'─'*col_w}")
    bets_flagged = sum(1 for r in recs if r.verdict == "BET")
    leans = sum(1 for r in recs if r.verdict == "LEAN")
    total_stake = sum(r.stake for r in recs if r.verdict in ("BET","LEAN"))
    print()
    kv("Games evaluated:", len(recs))
    kv("Bets flagged (★ BET ★):", bets_flagged)
    kv("Leans flagged:", leans)
    kv("Total stake today:", f"${total_stake:,.2f}")
    kv("Bankroll:", f"${bankroll:,.2f}")

    if bets_flagged == 0 and leans == 0:
        print()
        print("  No value found today. That's fine — discipline beats forcing bets.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Betting Model")
    parser.add_argument("--mode",     choices=["train","live","both"], default="both")
    parser.add_argument("--seasons",  nargs="+", type=int, default=CONFIG["seasons"])
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--min-edge", type=float, default=None)
    parser.add_argument("--kelly",    type=float, default=None)
    args = parser.parse_args()

    if args.bankroll: CONFIG["initial_bankroll"] = args.bankroll
    if args.min_edge: CONFIG["min_edge"]         = args.min_edge
    if args.kelly:    CONFIG["kelly_frac"]       = args.kelly
    if args.seasons:  CONFIG["seasons"]          = args.seasons

    print(f"\n{'═'*W}")
    print(f"  MLB BETTING MODEL")
    print(f"{'═'*W}")
    kv("Mode:",       args.mode)
    kv("Seasons:",    ", ".join(str(s) for s in CONFIG["seasons"]))
    kv("Bankroll:",   f"${CONFIG['initial_bankroll']:,.2f}")
    kv("Min edge:",   f"{CONFIG['min_edge']:.0%}")
    kv("Kelly frac:", f"{CONFIG['kelly_frac']:.0%}")

    model = None
    history_df = None

    if args.mode in ("train", "both"):
        model, history_df, games_raw = run_train(CONFIG["seasons"])

    if args.mode in ("live", "both"):
        if model is None:
            model = load_model()
            if model is None:
                print("  No saved model — run with --mode train first.")
                return
        if history_df is None:
            games_raw = load_games(CONFIG["seasons"])
            games_raw = load_odds_for_history(games_raw)
            team_stats = load_team_stats(CONFIG["seasons"])
            history_df = build_features(games_raw, team_stats)
        run_live(model, history_df)

    print(f"\n{'═'*W}")
    print("  DONE")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
