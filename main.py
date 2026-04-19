"""
main.py — MLB Betting Model

Usage:
    python main.py                              # train on last 3 seasons + current, then live picks
    python main.py --mode train                 # train only
    python main.py --mode live                  # live picks only (needs saved model)
    python main.py --bankroll 5000
    python main.py --min-edge 0.05
    python main.py --seasons 2022 2023 2024 2025

Bet types produced in live mode:
    - Moneyline        (The Odds API)
    - Run line spread  (The Odds API, -1.5 / +1.5)
    - 3-leg parlay     (top 3 moneyline edges combined)
"""

import os, argparse, json, time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from mlb_data import (
    fetch_season_schedule, fetch_team_season_stats,
    fetch_today_schedule, fetch_mlb_odds,
    fetch_pitcher_stats, fetch_lineup_stats,
    clear_corrupted_cache,
)
from mlb_features import build_features, build_game_features, TARGET_COL
import mlb_features as _feat_module
from mlb_model import (
    walk_forward_backtest, evaluate_model, train_final_model,
    save_model, load_model, predict_proba, get_feature_importance,
)
from mlb_betting import (
    evaluate_moneyline, evaluate_spread,
    build_parlay,
    run_backtest, edge_buckets, monthly_performance,
    american_to_decimal,
)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Default config ────────────────────────────────────────────────────────────
_CURRENT_YEAR = datetime.now().year
CONFIG = {
    # Train on last 3 completed seasons + current season
    "seasons":          [_CURRENT_YEAR - 3, _CURRENT_YEAR - 2,
                         _CURRENT_YEAR - 1, _CURRENT_YEAR],
    "initial_bankroll": 250.0,
    "min_edge":         0.06,   # 6% minimum edge — quality over quantity
    "kelly_frac":       0.25,
    "max_stake_pct":    0.03,    # hard cap: 3% of bankroll per bet
    "min_train_games":  300,
    "step_games":       30,
}

W = 112   # output width


# ─────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────

def section(title):
    print(f"\n{'─'*W}")
    print(f"  {title.upper()}")
    print(f"{'─'*W}")

def kv(label, value, w=34):
    print(f"  {label:<{w}} {value}")

def div():
    print(f"  {'─'*(W-4)}")


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_games(seasons):
    # Try exact cache file, then scan for any superset
    import glob
    candidates = [CACHE_DIR / f"mlb_games_{'_'.join(map(str,seasons))}.csv"]
    candidates += [Path(p) for p in sorted(glob.glob(str(CACHE_DIR / "mlb_games_*.csv")))]
    for cf in candidates:
        if not cf.exists():
            continue
        df = pd.read_csv(cf, parse_dates=["game_date"])
        df = df[df["game_date"].dt.year.isin(seasons)]
        if not df.empty:
            print(f"  Loaded {len(df)} games from {cf.name}. Delete to refresh.")
            return df

    print("  Fetching game history from MLB Stats API...")
    df = fetch_season_schedule(seasons)
    if not df.empty:
        fname = CACHE_DIR / f"mlb_games_{'_'.join(map(str,seasons))}.csv"
        df.to_csv(fname, index=False)
    return df


def load_team_stats(seasons):
    cache_file = CACHE_DIR / f"mlb_team_stats_{'_'.join(map(str,seasons))}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        print(f"  Team stats loaded from cache ({len(df)} rows).")
        return df
    dfs = []
    for s in seasons:
        try:
            dfs.append(fetch_team_season_stats(s))
            time.sleep(0.3)
        except Exception as e:
            print(f"  Warning: team stats {s}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(cache_file, index=False)
    return df


def load_odds_for_history(games_df):
    """Synthesise approximate historical odds for backtest simulation."""
    if "home_odds" in games_df.columns and games_df["home_odds"].notna().sum() > 100:
        return games_df

    print("  Synthesising historical odds from team win rates + park factors...")
    from mlb_data import PARK_FACTORS

    df = games_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    df["_roll_wr"] = (
        df.groupby("home_team")["home_win"]
        .transform(lambda x: x.shift(1).rolling(60, min_periods=10).mean())
        .fillna(0.54)
    )
    pf = df["home_team"].map(PARK_FACTORS).fillna(1.00)
    raw_prob  = ((df["_roll_wr"] * 0.65 + 0.54 * 0.35) * pf).clip(0.30, 0.72)
    rng       = np.random.default_rng(42)
    noisy_p   = (raw_prob + rng.normal(0, 0.02, len(df))).clip(0.28, 0.74)

    def to_am(p):
        p_vig = float(p) * 0.952
        p_vig = max(0.10, min(0.90, p_vig))
        if p_vig >= 0.5:
            return -int(round(p_vig / (1 - p_vig) * 100 / 5) * 5)
        else:
            return  int(round((1 - p_vig) / p_vig * 100 / 5) * 5)

    df["home_odds"] = noisy_p.apply(to_am)
    df["away_odds"] = (1 - noisy_p).apply(to_am)
    df = df.drop(columns=["_roll_wr"])
    return df


# ─────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────

def run_train(seasons):
    section("1 — loading game history")
    games_raw = load_games(seasons)
    if games_raw.empty:
        print("  ERROR: No game data. Check internet connection and run debug_api.py")
        return None, None, None

    games_raw = load_odds_for_history(games_raw)
    games_raw["game_date"] = pd.to_datetime(games_raw["game_date"])
    kv("Games loaded:",   len(games_raw))
    kv("Seasons:",        ", ".join(str(y) for y in sorted(games_raw["game_date"].dt.year.unique())))
    kv("Teams:",          games_raw["home_team"].nunique())
    kv("Home win rate:",  f"{games_raw['home_win'].mean():.1%}")

    section("2 — loading team stats")
    team_stats = load_team_stats(seasons)

    section("3 — building features")
    df = build_features(games_raw, team_stats if not team_stats.empty else None)
    FC = _feat_module.FEATURE_COLS
    n_valid = df.dropna(subset=FC).shape[0]
    kv("Complete feature rows:", n_valid)
    kv("Active features:",       len(FC))

    if n_valid < CONFIG["min_train_games"] + 50:
        print(f"\n  WARNING: Only {n_valid} complete rows. "
              f"Need {CONFIG['min_train_games']+50}+. Add more seasons.")

    section("4 — walk-forward model training")
    df = walk_forward_backtest(
        df, FC, TARGET_COL,
        min_train_games = CONFIG["min_train_games"],
        step_games      = CONFIG["step_games"],
    )
    n_pred = df["model_prob"].notna().sum()
    kv("Predictions generated:", n_pred)

    if n_pred == 0:
        print("  ERROR: No predictions. min_train_games may exceed dataset size.")
        return None, None, None

    section("5 — model quality")
    metrics = evaluate_model(df)
    kv("Games evaluated:",    metrics["n_predictions"])
    kv("Home win rate:",      f"{metrics['home_win_rate']:.1%}")
    kv("ROC-AUC:",            metrics["roc_auc"])
    kv("Brier skill score:",  metrics["brier_skill_score"])
    kv("Log loss:",           metrics["log_loss"])
    print()
    print("  Calibration (predicted → actual):")
    for pred, actual in metrics["calibration_curve"]:
        bar = "█" * int(actual * 20)
        diff = actual - pred
        print(f"    {pred:.2f} → {actual:.2f}  {bar:<20}  {'+' if diff>=0 else ''}{diff:.3f}")

    section("6 — betting backtest")
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
        div()
        kv("Total bets:",             summary["total_bets"])
        kv("Win / Loss:",             f"{summary['wins']} W  {summary['losses']} L")
        kv("Win rate:",               f"{summary['win_rate']:.1%}")
        kv("Avg edge:",               f"{summary['avg_edge_pct']}%")
        kv("Avg stake (fixed-base):", f"${summary['avg_stake']:.2f}")
        kv("Avg odds:",               f"{summary['avg_odds']:+.0f}")
        div()
        kv("Initial bankroll:",       f"${summary['initial_bankroll']:,.2f}")
        kv("Final bankroll (Kelly):", f"${summary['final_bankroll_kelly']:,.2f}")
        kv("Total P&L:",              f"${summary['total_pnl_kelly']:,.2f}")
        kv("ROI % (Kelly):",          f"{summary['roi_pct_kelly']}%")
        kv("Max drawdown:",           f"{summary['max_drawdown_pct']}%")
        kv("Profit factor:",          summary["profit_factor"])
        div()
        kv("Flat stake ROI %:",       f"{summary['flat_roi_pct']}%")
        kv("Flat final bankroll:",    f"${summary['flat_final_bankroll']:,.2f}")
        div()
        kv("Max win streak:",         summary["max_win_streak"])
        kv("Max loss streak:",        summary["max_loss_streak"])

        with open(CACHE_DIR / "last_summary.json", "w") as f:
            json.dump({k: v for k, v in summary.items()
                       if not isinstance(v, (list, dict))}, f)

        section("6a — edge bucket analysis")
        eb = edge_buckets(ledger)
        if not eb.empty:
            print(eb.to_string())

        section("6b — monthly performance")
        mp = monthly_performance(ledger)
        if not mp.empty:
            print(mp.to_string())

    section("7 — feature importance")
    final_model = train_final_model(df, FC, TARGET_COL)
    imp = get_feature_importance(final_model, FC)
    if not imp.empty:
        print("  Top 12 features:")
        for feat, score in imp.head(12).items():
            bar = "█" * int(score * 300)
            print(f"    {feat:<48} {score:.4f}  {bar}")

    save_model(final_model)
    return final_model, df, games_raw


# ─────────────────────────────────────────────
# Live picks
# ─────────────────────────────────────────────

def _build_upcoming_features(upcoming, history_df):
    """Build a feature dict for each upcoming game."""
    FC = _feat_module.FEATURE_COLS
    league_medians = {col: float(history_df[col].median())
                      for col in FC if col in history_df.columns
                      and history_df[col].notna().any()}
    games = []
    for _, game in upcoming.iterrows():
        home, away = game["home_team"], game["away_team"]
        feat = build_game_features(home, away, history_df)
        feat.update({
            "game_pk":   int(game.get("game_pk", 0)),
            "home_team": home,
            "away_team": away,
            "home_sp_id":   game.get("home_sp_id"),
            "home_sp_name": str(game.get("home_sp_name", "TBD")),
            "away_sp_id":   game.get("away_sp_id"),
            "away_sp_name": str(game.get("away_sp_name", "TBD")),
        })
        games.append(feat)
    return games


def _print_rec(rec, label_width=6):
    """Print a single bet recommendation row."""
    tag = {
        "BET":      "★ BET",
        "LEAN":     "~ lean",
        "SKIP":     "skip",
        "NO_VALUE": "——",
    }.get(rec.verdict, rec.verdict)
    edge_s  = f"{'+' if rec.edge >= 0 else ''}{rec.edge*100:.1f}%"
    stake_s = f"${rec.stake:.2f}" if rec.verdict in ("BET","LEAN") else "——"
    odds_s  = f"{int(rec.american_odds):+d}"

    if rec.bet_type == "parlay":
        print(f"\n  ┌─ 3-LEG PARLAY  {odds_s}  Prob:{rec.model_prob*100:.1f}%  "
              f"Edge:{edge_s}  Stake:{stake_s}  [{tag}]")
        for leg in rec.bet_side.split("\n    "):
            print(f"  │  {leg.strip()}")
        print(f"  └─ Note: {' | '.join(rec.notes)}")
    else:
        type_tag = {"moneyline": "ML", "spread": "RL"}.get(rec.bet_type, rec.bet_type.upper())
        matchup  = f"{rec.away_team} @ {rec.home_team}"
        # Resolve bet_side ("home"/"away") to actual team name
        if rec.bet_side in ("home", "home -1.5"):
            pick = rec.home_team
            line = "" if rec.bet_type == "moneyline" else " -1.5"
        elif rec.bet_side in ("away", "away +1.5"):
            pick = rec.away_team
            line = "" if rec.bet_type == "moneyline" else " +1.5"
        else:
            # spread side stored as e.g. "home -1.5" or "away +1.5"
            parts = rec.bet_side.split()
            pick  = rec.home_team if parts[0] == "home" else rec.away_team
            line  = " " + " ".join(parts[1:]) if len(parts) > 1 else ""
        pick_s = f"{pick}{line}"
        print(f"  {type_tag:<4} {matchup:<44}  PICK: {pick_s:<28} {odds_s:>6}  "
              f"Model:{rec.model_prob*100:>5.1f}%  Fair:{rec.fair_prob*100:>5.1f}%  "
              f"Edge:{edge_s:>6}  Stake:{stake_s:>8}  [{tag}]")
        if rec.notes:
            print(f"       {'':44}  {'':35} Note: {' | '.join(rec.notes)}")


def run_live(model, history_df):
    section("today's mlb picks")

    # Always use the configured bankroll for live picks.
    # Training results do NOT carry over — each session starts fresh.
    bankroll = CONFIG["initial_bankroll"]

    FC = _feat_module.FEATURE_COLS

    # ── Fetch schedule & odds ────────────────────────────────────────────
    today_df = fetch_today_schedule()
    if today_df.empty:
        print("  No MLB games today.")
        return

    odds_df   = fetch_mlb_odds()
    season    = datetime.now().year
    game_list = _build_upcoming_features(today_df, history_df)

    # ── Prepare output ───────────────────────────────────────────────────
    all_recs = []
    ml_recs  = []

    print()
    kv("Bankroll:", f"${bankroll:,.2f}")
    kv("Min edge:", f"{CONFIG['min_edge']:.0%}")
    kv("Max stake per bet:", f"{CONFIG['max_stake_pct']:.0%} of bankroll = "
       f"${bankroll * CONFIG['max_stake_pct']:.2f}")
    print()

    for feat in game_list:
        home = feat["home_team"]
        away = feat["away_team"]
        gpk  = feat["game_pk"]

        model_prob = predict_proba(model, feat, FC)

        # Fetch live SP stats
        home_sp_stats, away_sp_stats = {}, {}
        try:
            if feat.get("home_sp_id"):
                home_sp_stats = fetch_pitcher_stats(int(feat["home_sp_id"]), season) or {}
        except Exception:
            pass
        try:
            if feat.get("away_sp_id"):
                away_sp_stats = fetch_pitcher_stats(int(feat["away_sp_id"]), season) or {}
        except Exception:
            pass

        home_sp = feat.get("home_sp_name", "TBD")
        away_sp = feat.get("away_sp_name", "TBD")

        # Match odds row
        def match(df):
            if df.empty:
                return None
            mask = ((df["home_team"] == home) & (df["away_team"] == away))
            if not mask.any():
                # fuzzy match on last word of team name
                mask = (df["home_team"].str.contains(home.split()[-1], case=False, na=False) &
                        df["away_team"].str.contains(away.split()[-1], case=False, na=False))
            return df[mask].iloc[0] if mask.any() else None

        odds_row = match(odds_df)
        if odds_row is None:
            continue   # silently skip — no odds means game likely already started

        # ── Skip games that have already started ─────────────────────────
        # Check commence_time from odds API (UTC ISO string)
        commence = odds_row.get("commence_time", "")
        if commence:
            try:
                from datetime import timezone
                game_utc = datetime.fromisoformat(
                    commence.replace("Z", "+00:00")
                )
                now_utc = datetime.now(timezone.utc)
                if game_utc <= now_utc:
                    continue   # game has started or passed — skip
            except Exception:
                pass
        # Also check MLB status field from schedule (belt-and-suspenders)
        today_row = today_df[today_df["game_pk"] == gpk]
        if not today_row.empty:
            mlb_status = str(today_row.iloc[0].get("status", ""))
            if any(s in mlb_status for s in ("Live", "In Progress", "Manager", "Warmup")):
                continue

        n_books = int(odds_row.get("n_books", 1))

        # ── Moneyline ────────────────────────────────────────────────────
        ml = evaluate_moneyline(
            game_pk=gpk, home_team=home, away_team=away,
            home_sp=home_sp, away_sp=away_sp,
            model_prob_home=model_prob,
            home_odds=float(odds_row["home_odds"]),
            away_odds=float(odds_row["away_odds"]),
            initial_bankroll=bankroll,
            n_books=n_books, min_edge=CONFIG["min_edge"],
            kelly_frac=CONFIG["kelly_frac"],
            max_stake_pct=CONFIG["max_stake_pct"],
        )
        ml_recs.append(ml)
        all_recs.append(ml)

        # ── Run line ─────────────────────────────────────────────────────
        if pd.notna(odds_row.get("rl_home_odds")):
            rl = evaluate_spread(
                game_pk=gpk, home_team=home, away_team=away,
                home_sp=home_sp, away_sp=away_sp,
                model_prob_home=model_prob,
                rl_home_line=float(odds_row["rl_home_line"]),
                rl_home_odds=float(odds_row["rl_home_odds"]),
                rl_away_line=float(odds_row["rl_away_line"]),
                rl_away_odds=float(odds_row["rl_away_odds"]),
                initial_bankroll=bankroll,
                min_edge=CONFIG["min_edge"],
                kelly_frac=CONFIG["kelly_frac"],
                max_stake_pct=CONFIG["max_stake_pct"],
            )
            if rl:
                all_recs.append(rl)



    # ── 3-leg parlay — built from the top qualifying ML bets only ─────────
    # Pre-filter to BET-only before building parlay so we don't combine weak legs
    ml_bets_for_parlay = sorted(
        [r for r in ml_recs if r.verdict == "BET"],
        key=lambda x: x.edge, reverse=True
    )
    parlay = build_parlay(ml_bets_for_parlay, bankroll,
                          kelly_frac=0.10, max_stake_pct=0.02)
    if parlay:
        all_recs.append(parlay)

    # ── Print results — quality over quantity: top 3 per category, BET only ──
    section("moneyline picks")
    # Only show BET (not LEAN), sorted by edge, capped at 3
    ml_bets = sorted(
        [r for r in ml_recs if r.verdict == "BET"],
        key=lambda x: x.edge, reverse=True
    )[:3]
    if ml_bets:
        for r in ml_bets:
            _print_rec(r)
    else:
        print("  No high-confidence moneyline value found today.")
        print(f"  (Requires edge ≥ {CONFIG['min_edge']:.0%}. "
              f"Leans below threshold are filtered out.)")

    section("run line picks")
    rl_bets = sorted(
        [r for r in all_recs if r.bet_type == "spread" and r.verdict == "BET"],
        key=lambda x: x.edge, reverse=True
    )[:3]
    if rl_bets:
        for r in rl_bets:
            _print_rec(r)
    else:
        print("  No high-confidence run line value found today.")

    section("3-leg parlay recommendation")
    # Parlay is built from top moneyline bets only
    if parlay and parlay.verdict == "BET":
        _print_rec(parlay)
    elif len(ml_bets) < 3:
        print(f"  Need 3 qualifying moneyline picks to build a parlay "
              f"({len(ml_bets)} found today).")
    else:
        print("  Parlay edge below threshold — not recommended today.")

    # ── Summary ───────────────────────────────────────────────────────────
    section("daily summary")
    all_bets    = ml_bets + rl_bets
    total_stake = sum(r.stake for r in all_bets)
    if parlay and parlay.verdict == "BET":
        total_stake += parlay.stake
    kv("Games with odds available:", len(set(r.game_pk for r in all_recs if r.game_pk > 0)))
    kv("Moneyline bets:",  len(ml_bets))
    kv("Run line bets:",   len(rl_bets))
    kv("Parlay:",          "Yes" if parlay and parlay.verdict == "BET" else "No")
    kv("Total stake:",     f"${total_stake:.2f}")
    kv("Bankroll:",        f"${bankroll:,.2f}")
    kv("Edge threshold:",  f"{CONFIG['min_edge']:.0%} minimum (quality filter)")

    if not all_bets:
        print()
        print("  No value found today — discipline beats forcing bets.")

    if not bets and not leans:
        print()
        print("  No value found today — discipline beats forcing bets.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Betting Model")
    parser.add_argument("--mode",     choices=["train","live","both"], default="both")
    parser.add_argument("--seasons",  nargs="+", type=int, default=None)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--min-edge", type=float, default=None)
    parser.add_argument("--kelly",    type=float, default=None)
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete corrupted cache files and exit")
    args = parser.parse_args()

    if getattr(args, "clear_cache", False):
        clear_corrupted_cache()
        return

    if args.bankroll:  CONFIG["initial_bankroll"] = args.bankroll
    if args.min_edge:  CONFIG["min_edge"]         = args.min_edge
    if args.kelly:     CONFIG["kelly_frac"]       = args.kelly
    if args.seasons:   CONFIG["seasons"]          = args.seasons

    print(f"\n{'═'*W}")
    print(f"  MLB BETTING MODEL")
    print(f"{'═'*W}")
    kv("Mode:",         args.mode)
    kv("Seasons:",      ", ".join(str(s) for s in CONFIG["seasons"]))
    kv("Bankroll:",     f"${CONFIG['initial_bankroll']:,.2f}")
    kv("Min edge:",     f"{CONFIG['min_edge']:.0%}")
    kv("Kelly frac:",   f"{CONFIG['kelly_frac']:.0%}")
    kv("Max stake:",    f"{CONFIG['max_stake_pct']:.0%} per bet  "
                        f"= ${CONFIG['initial_bankroll'] * CONFIG['max_stake_pct']:.2f}")

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
            games_raw  = load_games(CONFIG["seasons"])
            games_raw  = load_odds_for_history(games_raw)
            team_stats = load_team_stats(CONFIG["seasons"])
            history_df = build_features(
                games_raw,
                team_stats if not team_stats.empty else None
            )
        run_live(model, history_df)

    print(f"\n{'═'*W}\n  DONE\n{'═'*W}\n")


if __name__ == "__main__":
    main()
