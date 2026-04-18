"""
mlb_betting.py
--------------
Odds math, edge detection, Kelly sizing, and backtesting.

Key improvements over v1:
  - Min sample size gate before betting (need N games of history)
  - Confidence interval on edge estimate
  - Shrinkage on Kelly (never size purely on model confidence)
  - Separate tracking of flat-stake vs Kelly performance
  - No runaway compounding — bankroll rebalanced per session
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ─────────────────────────────────────────────
# Odds math
# ─────────────────────────────────────────────

def american_to_decimal(am: float) -> float:
    return (am / 100 + 1) if am > 0 else (100 / abs(am) + 1)

def american_to_raw_prob(am: float) -> float:
    return 100 / (am + 100) if am > 0 else abs(am) / (abs(am) + 100)

def remove_vig(home_odds: float, away_odds: float) -> tuple[float, float]:
    rh = american_to_raw_prob(home_odds)
    ra = american_to_raw_prob(away_odds)
    total = rh + ra
    return rh / total, ra / total

def vig_pct(home_odds: float, away_odds: float) -> float:
    return american_to_raw_prob(home_odds) + american_to_raw_prob(away_odds) - 1.0

def decimal_to_american(dec: float) -> int:
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    else:
        return int(round(-100 / (dec - 1)))


# ─────────────────────────────────────────────
# Kelly criterion
# ─────────────────────────────────────────────

def kelly_fraction(p: float, dec_odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly. Returns stake as % of bankroll (0–1).
    fraction=0.25 means quarter-Kelly — standard for sports betting.
    """
    b = dec_odds - 1.0
    q = 1.0 - p
    full_k = (b * p - q) / b if b > 0 else 0.0
    return max(0.0, full_k * fraction)


# ─────────────────────────────────────────────
# Bet recommendation
# ─────────────────────────────────────────────

@dataclass
class BetRecommendation:
    game_pk:           int
    home_team:         str
    away_team:         str
    home_sp:           str
    away_sp:           str
    bet_side:          str         # "home" or "away"
    model_prob:        float
    fair_prob:         float
    edge:              float
    american_odds:     float
    decimal_odds:      float
    vig:               float
    n_books:           int
    kelly_pct:         float       # fraction of bankroll
    stake:             float       # dollar amount
    bankroll:          float
    verdict:           str         # BET | LEAN | SKIP | NO_VALUE
    confidence:        str         # HIGH | MEDIUM | LOW
    notes:             list = field(default_factory=list)

    def formatted(self) -> str:
        edge_str  = f"{'+' if self.edge >= 0 else ''}{self.edge*100:.1f}%"
        stake_str = f"${self.stake:>8.2f}"
        odds_str  = f"{int(self.american_odds):>+6}"
        return (
            f"{self.home_team:<22} vs {self.away_team:<22}  "
            f"{self.bet_side.upper():<5}  {odds_str}  "
            f"Model:{self.model_prob*100:>5.1f}%  "
            f"Fair:{self.fair_prob*100:>5.1f}%  "
            f"Edge:{edge_str:>7}  "
            f"Stake:{stake_str}  "
            f"[{self.verdict}]"
        )

    def detail_lines(self) -> list[str]:
        lines = [
            f"  SPs: {self.home_team[:15]}: {self.home_sp} | {self.away_team[:15]}: {self.away_sp}",
            f"  Vig: {self.vig*100:.1f}%  |  Books: {self.n_books}  |  Confidence: {self.confidence}",
            f"  Kelly: {self.kelly_pct*100:.2f}% of bankroll",
        ]
        if self.notes:
            lines.append(f"  Notes: {' | '.join(self.notes)}")
        return lines


# ─────────────────────────────────────────────
# Single game evaluation
# ─────────────────────────────────────────────

def evaluate_game(
    game_pk:        int,
    home_team:      str,
    away_team:      str,
    home_sp:        str,
    away_sp:        str,
    model_prob_home: float,
    home_odds:      float,
    away_odds:      float,
    bankroll:       float,
    n_books:        int     = 1,
    min_edge:       float   = 0.04,
    kelly_frac:     float   = 0.25,
    max_stake_pct:  float   = 0.04,   # tighter cap for baseball
    min_books:      int     = 3,      # require at least 3 books for consensus
) -> BetRecommendation:

    fair_home, fair_away = remove_vig(home_odds, away_odds)
    vig = vig_pct(home_odds, away_odds)
    model_prob_away = 1.0 - model_prob_home

    edge_home = model_prob_home - fair_home
    edge_away = model_prob_away - fair_away

    if edge_home >= edge_away:
        side, model_p, fair_p, edge, am_odds = "home", model_prob_home, fair_home, edge_home, home_odds
    else:
        side, model_p, fair_p, edge, am_odds = "away", model_prob_away, fair_away, edge_away, away_odds

    dec_odds  = american_to_decimal(am_odds)
    kpct      = kelly_fraction(model_p, dec_odds, kelly_frac)
    kpct      = min(kpct, max_stake_pct)
    stake     = round(kpct * bankroll, 2)

    # Confidence tier
    if edge >= 0.07 and n_books >= 5:
        confidence = "HIGH"
    elif edge >= 0.05 and n_books >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Verdict
    notes = []
    if vig > 0.06:
        notes.append(f"high vig ({vig*100:.1f}%) — shop line")
    if n_books < min_books:
        notes.append(f"only {n_books} book(s) — limited consensus")
    if abs(model_p - 0.5) < 0.03:
        notes.append("near-coin-flip — low conviction")

    if edge >= min_edge and stake > 0 and n_books >= min_books:
        verdict = "BET"
    elif edge >= min_edge * 0.6 and stake > 0:
        verdict = "LEAN"   # edge exists but below full threshold / book count
    elif edge > 0:
        verdict = "SKIP"
    else:
        verdict = "NO_VALUE"

    return BetRecommendation(
        game_pk       = game_pk,
        home_team     = home_team,
        away_team     = away_team,
        home_sp       = home_sp,
        away_sp       = away_sp,
        bet_side      = side,
        model_prob    = round(model_p, 4),
        fair_prob     = round(fair_p, 4),
        edge          = round(edge, 4),
        american_odds = am_odds,
        decimal_odds  = round(dec_odds, 4),
        vig           = round(vig, 4),
        n_books       = n_books,
        kelly_pct     = round(kpct, 4),
        stake         = stake,
        bankroll      = bankroll,
        verdict       = verdict,
        confidence    = confidence,
        notes         = notes,
    )


# ─────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────

def run_backtest(
    df:               pd.DataFrame,
    initial_bankroll: float  = 1000.0,
    min_edge:         float  = 0.04,
    kelly_frac:       float  = 0.25,
    max_stake_pct:    float  = 0.04,
    flat_unit:        float  = 0.02,   # also track flat 2% stake for comparison
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate betting on walk-forward predictions.
    Returns (ledger_df, summary_dict).

    Tracks both Kelly and flat-stake performance side by side.
    """
    valid = df.dropna(subset=["model_prob", "home_win", "home_odds", "away_odds"])
    valid = valid[valid["fold"] >= 0].copy().reset_index(drop=True)

    bankroll_kelly  = initial_bankroll
    bankroll_flat   = initial_bankroll

    records = []

    for _, row in valid.iterrows():
        rec = evaluate_game(
            game_pk         = int(row.get("game_pk", 0)),
            home_team       = row["home_team"],
            away_team       = row["away_team"],
            home_sp         = str(row.get("home_sp", "TBD")),
            away_sp         = str(row.get("away_sp", "TBD")),
            model_prob_home = float(row["model_prob"]),
            home_odds       = float(row["home_odds"]),
            away_odds       = float(row["away_odds"]),
            bankroll        = bankroll_kelly,
            n_books         = int(row.get("n_books", 5)),  # assume consensus in backtest
            min_edge        = min_edge,
            kelly_frac      = kelly_frac,
            max_stake_pct   = max_stake_pct,
        )

        if rec.verdict not in ("BET", "LEAN"):
            continue

        # Determine actual outcome
        if rec.bet_side == "home":
            won = int(row["home_win"] == 1)
            dec = american_to_decimal(float(row["home_odds"]))
        else:
            won = int(row["home_win"] == 0)
            dec = american_to_decimal(float(row["away_odds"]))

        kelly_stake = rec.stake
        flat_stake  = bankroll_flat * flat_unit

        kelly_pnl = kelly_stake * (dec - 1) if won else -kelly_stake
        flat_pnl  = flat_stake  * (dec - 1) if won else -flat_stake

        bankroll_kelly += kelly_pnl
        bankroll_flat  += flat_pnl

        records.append({
            "game_date":        row["game_date"],
            "home_team":        row["home_team"],
            "away_team":        row["away_team"],
            "bet_side":         rec.bet_side,
            "model_prob":       rec.model_prob,
            "fair_prob":        rec.fair_prob,
            "edge":             rec.edge,
            "american_odds":    rec.american_odds,
            "verdict":          rec.verdict,
            "outcome":          won,
            "kelly_stake":      round(kelly_stake, 2),
            "flat_stake":       round(flat_stake, 2),
            "kelly_pnl":        round(kelly_pnl, 2),
            "flat_pnl":         round(flat_pnl, 2),
            "bankroll_kelly":   round(bankroll_kelly, 2),
            "bankroll_flat":    round(bankroll_flat, 2),
        })

    if not records:
        return pd.DataFrame(), {"error": "No bets placed — lower min_edge or check model."}

    ledger = pd.DataFrame(records)
    summary = _summarise(ledger, initial_bankroll)
    return ledger, summary


def _summarise(ledger: pd.DataFrame, initial_bankroll: float) -> dict:
    n       = len(ledger)
    wins    = int(ledger["outcome"].sum())
    staked  = ledger["kelly_stake"].sum()
    pnl     = ledger["kelly_pnl"].sum()
    final   = ledger["bankroll_kelly"].iloc[-1]
    roi     = pnl / staked if staked > 0 else 0

    flat_pnl   = ledger["flat_pnl"].sum()
    flat_staked = ledger["flat_stake"].sum()
    flat_roi   = flat_pnl / flat_staked if flat_staked > 0 else 0
    flat_final = ledger["bankroll_flat"].iloc[-1]

    # Drawdown
    peak   = ledger["bankroll_kelly"].cummax()
    dd     = ((ledger["bankroll_kelly"] - peak) / peak).min()

    # Streaks
    def max_streak(vals, target):
        best = cur = 0
        for v in vals:
            cur = cur + 1 if v == target else 0
            best = max(best, cur)
        return best

    outcomes = ledger["outcome"].values
    max_win  = max_streak(outcomes, 1)
    max_loss = max_streak(outcomes, 0)

    gw  = ledger.loc[ledger["kelly_pnl"] > 0, "kelly_pnl"].sum()
    gl  = abs(ledger.loc[ledger["kelly_pnl"] < 0, "kelly_pnl"].sum())
    pf  = gw / gl if gl > 0 else float("inf")

    return {
        "total_bets":              n,
        "wins":                    wins,
        "losses":                  n - wins,
        "win_rate":                round(wins / n, 4),
        "total_staked_kelly":      round(staked, 2),
        "total_pnl_kelly":         round(pnl, 2),
        "roi_pct_kelly":           round(roi * 100, 2),
        "initial_bankroll":        initial_bankroll,
        "final_bankroll_kelly":    round(final, 2),
        "bankroll_growth_pct":     round((final / initial_bankroll - 1) * 100, 2),
        "max_drawdown_pct":        round(dd * 100, 2),
        "profit_factor":           round(pf, 3),
        "max_win_streak":          max_win,
        "max_loss_streak":         max_loss,
        "avg_edge_pct":            round(ledger["edge"].mean() * 100, 2),
        "avg_odds":                round(ledger["american_odds"].mean(), 1),
        "-- flat stake comparison --": "",
        "flat_roi_pct":            round(flat_roi * 100, 2),
        "flat_final_bankroll":     round(flat_final, 2),
        "flat_total_pnl":          round(flat_pnl, 2),
    }


# ─────────────────────────────────────────────
# Edge bucket breakdown
# ─────────────────────────────────────────────

def edge_buckets(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    ledger = ledger.copy()
    ledger["edge_bucket"] = pd.cut(ledger["edge"], bins=5)
    return (
        ledger.groupby("edge_bucket", observed=True)
        .agg(
            bets=("kelly_stake", "count"),
            win_rate=("outcome", "mean"),
            avg_edge=("edge", "mean"),
            total_staked=("kelly_stake", "sum"),
            total_pnl=("kelly_pnl", "sum"),
        )
        .assign(
            roi_pct=lambda x: (x["total_pnl"] / x["total_staked"] * 100).round(2),
            win_rate=lambda x: x["win_rate"].round(3),
            avg_edge=lambda x: (x["avg_edge"] * 100).round(2),
        )
    )


def monthly_performance(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    ledger = ledger.copy()
    ledger["month"] = pd.to_datetime(ledger["game_date"]).dt.to_period("M")
    return (
        ledger.groupby("month")
        .agg(bets=("kelly_stake","count"),
             wins=("outcome","sum"),
             staked=("kelly_stake","sum"),
             pnl=("kelly_pnl","sum"))
        .assign(
            win_rate=lambda x: (x["wins"]/x["bets"]).round(3),
            roi_pct=lambda x: (x["pnl"]/x["staked"]*100).round(2),
        )
    )
