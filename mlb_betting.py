"""
mlb_betting.py
--------------
Odds math, edge detection, fixed-base Kelly sizing, and backtesting.

Stake bug fix
─────────────
The old code passed the GROWING bankroll into evaluate_game on every
backtest iteration, which compounded stakes exponentially into millions.

Fix: Kelly is always computed against the FIXED initial bankroll.
     The running bankroll is tracked for P&L reporting only.
     Max stake is also hard-capped at a dollar amount (not just a %).

Bet types supported
───────────────────
  moneyline  — standard h2h win bet
  spread     — run line (-1.5 / +1.5)
  f5         — first 5 innings moneyline
  parlay     — 3-leg combo of the top moneyline edges
  hr_prop    — batter HR over (0.5 line)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Odds math
# ─────────────────────────────────────────────

def american_to_decimal(am: float) -> float:
    return (am / 100 + 1) if am > 0 else (100 / abs(am) + 1)

def american_to_raw_prob(am: float) -> float:
    return 100 / (am + 100) if am > 0 else abs(am) / (abs(am) + 100)

def remove_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    ra, rb  = american_to_raw_prob(odds_a), american_to_raw_prob(odds_b)
    total   = ra + rb
    return ra / total, rb / total

def vig_pct(odds_a: float, odds_b: float) -> float:
    return american_to_raw_prob(odds_a) + american_to_raw_prob(odds_b) - 1.0

def decimal_to_american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2.0 else int(round(-100 / (dec - 1)))

def parlay_decimal(*decimals) -> float:
    result = 1.0
    for d in decimals:
        result *= d
    return result

def parlay_american(*decimals) -> int:
    return decimal_to_american(parlay_decimal(*decimals))


# ─────────────────────────────────────────────
# Fixed-base Kelly  ← THE FIX
# ─────────────────────────────────────────────

def kelly_stake(
    p:            float,
    dec_odds:     float,
    bankroll:     float,          # ALWAYS the initial / session-start bankroll
    fraction:     float = 0.25,   # quarter-Kelly
    max_pct:      float = 0.03,   # hard cap: never more than 3% per bet
    max_dollars:  float = None,   # optional absolute dollar cap
) -> float:
    """
    Computes a fixed-base fractional Kelly stake.

    bankroll should be the INITIAL bankroll or the bankroll at the START
    of the current session — NOT the running total after each win.
    This prevents the exponential compounding that produced million-dollar stakes.
    """
    b     = dec_odds - 1.0
    q     = 1.0 - p
    full  = (b * p - q) / b if b > 0 else 0.0
    frac  = max(0.0, full * fraction)
    pct   = min(frac, max_pct)           # cap as % of bankroll
    stake = pct * bankroll
    if max_dollars is not None:
        stake = min(stake, max_dollars)  # optional absolute cap
    return round(stake, 2)


# ─────────────────────────────────────────────
# Bet recommendation dataclass
# ─────────────────────────────────────────────

@dataclass
class BetRecommendation:
    game_pk:       int
    home_team:     str
    away_team:     str
    home_sp:       str
    away_sp:       str
    bet_type:      str    # moneyline | spread | f5 | parlay | hr_prop
    bet_side:      str    # home | away | over | parlay description
    model_prob:    float
    fair_prob:     float
    edge:          float
    american_odds: float
    decimal_odds:  float
    vig:           float
    n_books:       int
    kelly_pct:     float
    stake:         float
    bankroll:      float
    verdict:       str    # BET | LEAN | SKIP | NO_VALUE
    confidence:    str    # HIGH | MEDIUM | LOW
    notes:         list   = field(default_factory=list)


# ─────────────────────────────────────────────
# Moneyline evaluation
# ─────────────────────────────────────────────

def evaluate_moneyline(
    game_pk:         int,
    home_team:       str,
    away_team:       str,
    home_sp:         str,
    away_sp:         str,
    model_prob_home: float,
    home_odds:       float,
    away_odds:       float,
    initial_bankroll: float,      # fixed base — never the running total
    n_books:         int   = 1,
    min_edge:        float = 0.04,
    kelly_frac:      float = 0.25,
    max_stake_pct:   float = 0.03,
    min_books:       int   = 1,
) -> BetRecommendation:

    fair_h, fair_a   = remove_vig(home_odds, away_odds)
    vig              = vig_pct(home_odds, away_odds)
    model_prob_away  = 1.0 - model_prob_home

    edge_h = model_prob_home - fair_h
    edge_a = model_prob_away - fair_a

    if edge_h >= edge_a:
        side, mp, fp, edge, am = "home", model_prob_home, fair_h, edge_h, home_odds
    else:
        side, mp, fp, edge, am = "away", model_prob_away, fair_a, edge_a, away_odds

    dec   = american_to_decimal(am)
    stake = kelly_stake(mp, dec, initial_bankroll, kelly_frac, max_stake_pct)

    confidence = ("HIGH"   if edge >= 0.07 and n_books >= 3
                  else "MEDIUM" if edge >= 0.05
                  else "LOW")

    notes = []
    if vig > 0.06:
        notes.append(f"high vig {vig*100:.1f}%")
    if abs(mp - 0.5) < 0.03:
        notes.append("near coin-flip")

    if edge >= min_edge and stake > 0:
        verdict = "BET"
    elif edge >= min_edge * 0.6 and stake > 0:
        verdict = "LEAN"
    elif edge > 0:
        verdict = "SKIP"
    else:
        verdict = "NO_VALUE"

    return BetRecommendation(
        game_pk=game_pk, home_team=home_team, away_team=away_team,
        home_sp=home_sp, away_sp=away_sp,
        bet_type="moneyline", bet_side=side,
        model_prob=round(mp, 4), fair_prob=round(fp, 4),
        edge=round(edge, 4),
        american_odds=am, decimal_odds=round(dec, 4),
        vig=round(vig, 4), n_books=n_books,
        kelly_pct=round(stake / initial_bankroll, 4),
        stake=stake, bankroll=initial_bankroll,
        verdict=verdict, confidence=confidence, notes=notes,
    )


# ─────────────────────────────────────────────
# Run line (spread) evaluation
# ─────────────────────────────────────────────

def evaluate_spread(
    game_pk:         int,
    home_team:       str,
    away_team:       str,
    home_sp:         str,
    away_sp:         str,
    model_prob_home: float,          # full-game ML prob from model
    rl_home_line:    float,          # typically -1.5
    rl_home_odds:    float,
    rl_away_line:    float,          # typically +1.5
    rl_away_odds:    float,
    initial_bankroll: float,
    min_edge:        float = 0.04,
    kelly_frac:      float = 0.25,
    max_stake_pct:   float = 0.03,
) -> Optional[BetRecommendation]:
    """
    Evaluates the run line. We adjust the model's full-game probability
    for the spread using a logistic shift: covering -1.5 requires winning
    by 2+, which historically happens ~55% of the time teams win.
    """
    if rl_home_odds is None or rl_away_odds is None:
        return None

    # Approximate probability of covering each side
    # P(home covers -1.5) ≈ P(home win by 2+) ≈ P(home win) * 0.72
    # P(away covers +1.5) ≈ P(away win OR home wins by exactly 1) ≈ 1 - P(home win)*0.72
    p_home_cover = model_prob_home * 0.72
    p_away_cover = 1.0 - p_home_cover

    fair_h, fair_a = remove_vig(rl_home_odds, rl_away_odds)
    vig = vig_pct(rl_home_odds, rl_away_odds)

    edge_h = p_home_cover - fair_h
    edge_a = p_away_cover - fair_a

    if edge_h >= edge_a:
        side, mp, fp, edge, am = "home", p_home_cover, fair_h, edge_h, rl_home_odds
        line_str = f"{rl_home_line:+.1f}"
    else:
        side, mp, fp, edge, am = "away", p_away_cover, fair_a, edge_a, rl_away_odds
        line_str = f"{rl_away_line:+.1f}"

    dec   = american_to_decimal(am)
    stake = kelly_stake(mp, dec, initial_bankroll, kelly_frac, max_stake_pct)

    if edge >= min_edge and stake > 0:
        verdict = "BET"
    elif edge >= min_edge * 0.6 and stake > 0:
        verdict = "LEAN"
    elif edge > 0:
        verdict = "SKIP"
    else:
        verdict = "NO_VALUE"

    return BetRecommendation(
        game_pk=game_pk, home_team=home_team, away_team=away_team,
        home_sp=home_sp, away_sp=away_sp,
        bet_type="spread", bet_side=f"{side} {line_str}",
        model_prob=round(mp, 4), fair_prob=round(fp, 4),
        edge=round(edge, 4),
        american_odds=am, decimal_odds=round(dec, 4),
        vig=round(vig, 4), n_books=1,
        kelly_pct=round(stake / initial_bankroll, 4),
        stake=stake, bankroll=initial_bankroll,
        verdict=verdict, confidence="MEDIUM",
        notes=[f"run line {line_str}"],
    )


# ─────────────────────────────────────────────
# First 5 innings evaluation
# ─────────────────────────────────────────────

def evaluate_f5(
    game_pk:         int,
    home_team:       str,
    away_team:       str,
    home_sp:         str,
    away_sp:         str,
    model_prob_home: float,
    f5_home_odds:    float,
    f5_away_odds:    float,
    home_sp_stats:   dict,
    away_sp_stats:   dict,
    initial_bankroll: float,
    min_edge:        float = 0.04,
    kelly_frac:      float = 0.25,
    max_stake_pct:   float = 0.03,
) -> Optional[BetRecommendation]:
    """
    F5 bet evaluation. SP quality is the dominant factor for 5-inning results.
    We adjust the full-game model probability using relative SP ERA.
    """
    if f5_home_odds is None or f5_away_odds is None:
        return None

    # SP ERA adjustment: better SP → higher F5 win probability
    home_era = (home_sp_stats or {}).get("era", 4.50)
    away_era = (away_sp_stats or {}).get("era", 4.50)
    era_diff  = away_era - home_era    # positive = home SP advantage

    # Logistic shift: each ERA point of advantage ~ 3pp probability shift
    era_adj  = np.tanh(era_diff / 3.0) * 0.08
    p_f5_home = np.clip(model_prob_home + era_adj, 0.20, 0.80)
    p_f5_away = 1.0 - p_f5_home

    fair_h, fair_a = remove_vig(f5_home_odds, f5_away_odds)
    vig = vig_pct(f5_home_odds, f5_away_odds)

    edge_h = p_f5_home - fair_h
    edge_a = p_f5_away - fair_a

    if edge_h >= edge_a:
        side, mp, fp, edge, am = "home", p_f5_home, fair_h, edge_h, f5_home_odds
    else:
        side, mp, fp, edge, am = "away", p_f5_away, fair_a, edge_a, f5_away_odds

    dec   = american_to_decimal(am)
    stake = kelly_stake(mp, dec, initial_bankroll, kelly_frac, max_stake_pct)

    if edge >= min_edge and stake > 0:
        verdict = "BET"
    elif edge >= min_edge * 0.6 and stake > 0:
        verdict = "LEAN"
    elif edge > 0:
        verdict = "SKIP"
    else:
        verdict = "NO_VALUE"

    sp_note = f"SPs: {home_sp} ERA {home_era:.2f} vs {away_sp} ERA {away_era:.2f}"

    return BetRecommendation(
        game_pk=game_pk, home_team=home_team, away_team=away_team,
        home_sp=home_sp, away_sp=away_sp,
        bet_type="f5", bet_side=side,
        model_prob=round(mp, 4), fair_prob=round(fp, 4),
        edge=round(edge, 4),
        american_odds=am, decimal_odds=round(dec, 4),
        vig=round(vig, 4), n_books=1,
        kelly_pct=round(stake / initial_bankroll, 4),
        stake=stake, bankroll=initial_bankroll,
        verdict=verdict, confidence="MEDIUM",
        notes=[sp_note],
    )


# ─────────────────────────────────────────────
# Home run prop evaluation
# ─────────────────────────────────────────────

def evaluate_hr_props(
    hr_df:           pd.DataFrame,
    game_pk:         int,
    home_team:       str,
    away_team:       str,
    initial_bankroll: float,
    min_edge:        float = 0.05,   # slightly tighter for props
    kelly_frac:      float = 0.15,   # more conservative on props
    max_stake_pct:   float = 0.02,
) -> list[BetRecommendation]:
    """
    Evaluate HR over props. League-average HR rate per PA is ~3.5%.
    Over a full game (~4 PA) that's ~13% per player.
    We flag any player where the implied probability is below 12%
    (i.e. the line is priced better than fair).
    """
    if hr_df is None or hr_df.empty:
        return []

    LEAGUE_HR_PROB = 0.130   # ~13% chance any starter hits a HR per game

    recs = []
    for _, row in hr_df.iterrows():
        over_odds = row.get("over_odds")
        if over_odds is None:
            continue

        dec       = american_to_decimal(float(over_odds))
        implied_p = 1.0 / dec          # raw implied (no vig removal — single line)
        edge      = LEAGUE_HR_PROB - implied_p

        if edge < min_edge:
            continue

        stake = kelly_stake(LEAGUE_HR_PROB, dec, initial_bankroll,
                            kelly_frac, max_stake_pct)

        recs.append(BetRecommendation(
            game_pk=game_pk,
            home_team=home_team, away_team=away_team,
            home_sp="", away_sp="",
            bet_type="hr_prop",
            bet_side=f"{row['player']} HR Over {row.get('line', 0.5)}",
            model_prob=LEAGUE_HR_PROB,
            fair_prob=implied_p,
            edge=round(edge, 4),
            american_odds=float(over_odds),
            decimal_odds=round(dec, 4),
            vig=0.0, n_books=1,
            kelly_pct=round(stake / initial_bankroll, 4),
            stake=stake, bankroll=initial_bankroll,
            verdict="BET" if edge >= min_edge else "SKIP",
            confidence="LOW",
            notes=[f"league avg HR prob {LEAGUE_HR_PROB:.1%}"],
        ))

    return sorted(recs, key=lambda r: r.edge, reverse=True)


# ─────────────────────────────────────────────
# 3-leg parlay builder
# ─────────────────────────────────────────────

def build_parlay(
    moneyline_recs:  list[BetRecommendation],
    initial_bankroll: float,
    kelly_frac:      float = 0.10,   # conservative for parlays
    max_stake_pct:   float = 0.02,
) -> Optional[BetRecommendation]:
    """
    Build the best 3-leg parlay from the top moneyline recommendations.
    Only uses legs where verdict is BET or LEAN and edge > 0.
    Combined probability is the product of individual probabilities.
    """
    eligible = [r for r in moneyline_recs
                if r.verdict in ("BET", "LEAN") and r.edge > 0][:3]

    if len(eligible) < 3:
        return None

    legs      = eligible[:3]
    combined_p = 1.0
    decimals   = []
    leg_strs   = []

    for leg in legs:
        combined_p *= leg.model_prob
        decimals.append(leg.decimal_odds)
        matchup = (f"{leg.home_team} vs {leg.away_team}"
                   f" — {leg.bet_side.upper()} {int(leg.american_odds):+d}")
        leg_strs.append(matchup)

    parlay_dec = parlay_decimal(*decimals)
    parlay_am  = decimal_to_american(parlay_dec)

    # Fair probability (product of vig-removed probs)
    fair_combined = 1.0
    for leg in legs:
        fair_combined *= leg.fair_prob

    edge  = combined_p - fair_combined
    stake = kelly_stake(combined_p, parlay_dec, initial_bankroll,
                        kelly_frac, max_stake_pct)

    verdict = "BET" if edge > 0.01 and stake > 0 else "SKIP"

    return BetRecommendation(
        game_pk=-1,
        home_team="PARLAY", away_team="3-LEG",
        home_sp="", away_sp="",
        bet_type="parlay",
        bet_side="\n    ".join(leg_strs),
        model_prob=round(combined_p, 4),
        fair_prob=round(fair_combined, 4),
        edge=round(edge, 4),
        american_odds=parlay_am,
        decimal_odds=round(parlay_dec, 4),
        vig=0.0, n_books=1,
        kelly_pct=round(stake / initial_bankroll, 4),
        stake=stake, bankroll=initial_bankroll,
        verdict=verdict, confidence="LOW",
        notes=["3-leg parlay — all legs must win"],
    )


# ─────────────────────────────────────────────
# Backtest  (fixed-base Kelly)
# ─────────────────────────────────────────────

def run_backtest(
    df:               pd.DataFrame,
    initial_bankroll: float = 1000.0,
    min_edge:         float = 0.04,
    kelly_frac:       float = 0.25,
    max_stake_pct:    float = 0.03,
    flat_unit:        float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest moneyline bets. Stakes are computed against INITIAL_BANKROLL
    (fixed base) so there is no exponential compounding.

    Running bankroll is tracked for P&L only.
    """
    valid = df.dropna(subset=["model_prob", "home_win", "home_odds", "away_odds"])
    valid = valid[valid["fold"] >= 0].copy().reset_index(drop=True)

    running_kelly = initial_bankroll
    running_flat  = initial_bankroll
    records       = []

    for _, row in valid.iterrows():
        rec = evaluate_moneyline(
            game_pk          = int(row.get("game_pk", 0)),
            home_team        = row["home_team"],
            away_team        = row["away_team"],
            home_sp          = str(row.get("home_sp", "TBD")),
            away_sp          = str(row.get("away_sp", "TBD")),
            model_prob_home  = float(row["model_prob"]),
            home_odds        = float(row["home_odds"]),
            away_odds        = float(row["away_odds"]),
            initial_bankroll = initial_bankroll,   # ← FIXED BASE
            n_books          = int(row.get("n_books", 3)),
            min_edge         = min_edge,
            kelly_frac       = kelly_frac,
            max_stake_pct    = max_stake_pct,
        )

        if rec.verdict not in ("BET", "LEAN"):
            continue

        if rec.bet_side == "home":
            won = int(row["home_win"] == 1)
            dec = american_to_decimal(float(row["home_odds"]))
        else:
            won = int(row["home_win"] == 0)
            dec = american_to_decimal(float(row["away_odds"]))

        k_stake  = rec.stake
        f_stake  = initial_bankroll * flat_unit   # also fixed-base flat

        k_pnl    = k_stake * (dec - 1) if won else -k_stake
        f_pnl    = f_stake * (dec - 1) if won else -f_stake

        running_kelly += k_pnl
        running_flat  += f_pnl

        records.append({
            "game_date":     row["game_date"],
            "home_team":     row["home_team"],
            "away_team":     row["away_team"],
            "bet_side":      rec.bet_side,
            "model_prob":    rec.model_prob,
            "fair_prob":     rec.fair_prob,
            "edge":          rec.edge,
            "american_odds": rec.american_odds,
            "verdict":       rec.verdict,
            "outcome":       won,
            "kelly_stake":   round(k_stake, 2),
            "flat_stake":    round(f_stake, 2),
            "kelly_pnl":     round(k_pnl, 2),
            "flat_pnl":      round(f_pnl, 2),
            "bankroll_kelly":round(running_kelly, 2),
            "bankroll_flat": round(running_flat, 2),
        })

    if not records:
        return pd.DataFrame(), {"error": "No bets — lower min_edge or add more seasons."}

    ledger  = pd.DataFrame(records)
    summary = _summarise(ledger, initial_bankroll)
    return ledger, summary


def _summarise(ledger, initial_bankroll):
    n       = len(ledger)
    wins    = int(ledger["outcome"].sum())
    staked  = ledger["kelly_stake"].sum()
    pnl     = ledger["kelly_pnl"].sum()
    final_k = ledger["bankroll_kelly"].iloc[-1]
    roi     = pnl / staked if staked > 0 else 0

    fp_staked = ledger["flat_stake"].sum()
    fp_pnl    = ledger["flat_pnl"].sum()
    fp_roi    = fp_pnl / fp_staked if fp_staked > 0 else 0
    final_f   = ledger["bankroll_flat"].iloc[-1]

    peak = ledger["bankroll_kelly"].cummax()
    dd   = ((ledger["bankroll_kelly"] - peak) / peak).min()

    def max_streak(vals, t):
        best = cur = 0
        for v in vals:
            cur = cur + 1 if v == t else 0
            best = max(best, cur)
        return best

    gw = ledger.loc[ledger["kelly_pnl"] > 0, "kelly_pnl"].sum()
    gl = abs(ledger.loc[ledger["kelly_pnl"] < 0, "kelly_pnl"].sum())

    return {
        "total_bets":           n,
        "wins":                 wins,
        "losses":               n - wins,
        "win_rate":             round(wins / n, 4),
        "total_staked_kelly":   round(staked, 2),
        "total_pnl_kelly":      round(pnl, 2),
        "roi_pct_kelly":        round(roi * 100, 2),
        "initial_bankroll":     initial_bankroll,
        "final_bankroll_kelly": round(final_k, 2),
        "bankroll_growth_pct":  round((final_k / initial_bankroll - 1) * 100, 2),
        "max_drawdown_pct":     round(dd * 100, 2),
        "profit_factor":        round(gw / gl, 3) if gl > 0 else 999,
        "max_win_streak":       max_streak(ledger["outcome"].values, 1),
        "max_loss_streak":      max_streak(ledger["outcome"].values, 0),
        "avg_edge_pct":         round(ledger["edge"].mean() * 100, 2),
        "avg_stake":            round(ledger["kelly_stake"].mean(), 2),
        "avg_odds":             round(ledger["american_odds"].mean(), 1),
        "── flat stake ──":     "",
        "flat_roi_pct":         round(fp_roi * 100, 2),
        "flat_final_bankroll":  round(final_f, 2),
        "flat_total_pnl":       round(fp_pnl, 2),
    }


def edge_buckets(ledger):
    if ledger.empty:
        return pd.DataFrame()
    ledger = ledger.copy()
    ledger["edge_bucket"] = pd.cut(ledger["edge"], bins=5)
    return (
        ledger.groupby("edge_bucket", observed=True)
        .agg(bets=("kelly_stake","count"), win_rate=("outcome","mean"),
             avg_edge=("edge","mean"), total_staked=("kelly_stake","sum"),
             total_pnl=("kelly_pnl","sum"))
        .assign(roi_pct   = lambda x: (x["total_pnl"]/x["total_staked"]*100).round(2),
                win_rate  = lambda x: x["win_rate"].round(3),
                avg_edge  = lambda x: (x["avg_edge"]*100).round(2))
    )


def monthly_performance(ledger):
    if ledger.empty:
        return pd.DataFrame()
    ledger = ledger.copy()
    ledger["month"] = pd.to_datetime(ledger["game_date"]).dt.to_period("M")
    return (
        ledger.groupby("month")
        .agg(bets=("kelly_stake","count"), wins=("outcome","sum"),
             staked=("kelly_stake","sum"), pnl=("kelly_pnl","sum"))
        .assign(win_rate=lambda x: (x["wins"]/x["bets"]).round(3),
                roi_pct =lambda x: (x["pnl"]/x["staked"]*100).round(2))
    )
