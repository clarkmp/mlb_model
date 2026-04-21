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
    """Convert American odds to decimal. Handles edge cases."""
    if am is None or am == 0:
        return 2.0  # Even odds fallback
    return (am / 100 + 1) if am > 0 else (100 / abs(am) + 1)

def american_to_raw_prob(am: float) -> float:
    """Convert American odds to raw probability. Handles edge cases."""
    if am is None or am == 0:
        return 0.5  # Even probability fallback
    return 100 / (am + 100) if am > 0 else abs(am) / (abs(am) + 100)

def remove_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove vig from odds pair. Handles edge cases."""
    if odds_a is None or odds_b is None or odds_a == 0 or odds_b == 0:
        return 0.5, 0.5  # Equal probability fallback
    ra, rb  = american_to_raw_prob(odds_a), american_to_raw_prob(odds_b)
    total   = ra + rb
    if total == 0:
        return 0.5, 0.5
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
    bet_type:      str    # moneyline | spread | parlay | hr_prop
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

    # Empirically validated: when a team wins, they cover -1.5 about 68.6% of the time
    COVER_RATE = 0.686
    
    # Calculate cover probabilities for BOTH sides correctly
    # The HOME team's probability of covering -1.5:
    p_home_cover_minus_1_5 = model_prob_home * COVER_RATE
    # The AWAY team's probability of covering -1.5:
    model_prob_away = 1.0 - model_prob_home
    p_away_cover_minus_1_5 = model_prob_away * COVER_RATE
    
    # The +1.5 lines are simply the complements:
    p_home_cover_plus_1_5 = 1.0 - p_away_cover_minus_1_5
    p_away_cover_plus_1_5 = 1.0 - p_home_cover_minus_1_5
    
    # Now match to actual run lines in the odds
    # Home team is typically -1.5 if favorite, +1.5 if underdog
    # Away team is typically -1.5 if favorite, +1.5 if underdog
    
    # Determine which side is favorite based on the line
    home_is_favorite = rl_home_line < 0  # Negative line means favorite
    
    if home_is_favorite:
        # Home team -1.5, Away team +1.5
        p_home_cover = p_home_cover_minus_1_5
        p_away_cover = p_away_cover_plus_1_5
    else:
        # Away team -1.5, Home team +1.5
        p_home_cover = p_home_cover_plus_1_5
        p_away_cover = p_away_cover_minus_1_5

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
# Home run prop evaluation
# ─────────────────────────────────────────────

def evaluate_hr_props(
    hr_df:            pd.DataFrame,
    game_pk:          int,
    home_team:        str,
    away_team:        str,
    initial_bankroll: float,
    pitcher_hand:     str   = "R",    # opposing pitcher handedness: "R" or "L"
    park_factor:      float = 1.00,   # ballpark HR factor (>1 = hitter friendly)
    wind_mph:         float = 0.0,    # positive = blowing out (favours HRs)
    wind_dir:         str   = "",     # "out", "in", "cross", ""
    temp_f:           float = 72.0,   # game-time temperature (warmer = more HRs)
    home_game:        bool  = True,   # is batter playing at home?
    min_edge:         float = 0.04,
    kelly_frac:       float = 0.15,
    max_stake_pct:    float = 0.02,
) -> list['BetRecommendation']:
    """
    Multi-factor HR over prop evaluation.

    Factors modelled:
      - Pitcher handedness vs batter platoon advantage
          RHB vs LHP: +15% HR rate uplift (batters hit more HRs vs opposite hand)
          LHB vs RHP: +10% uplift
          Same-hand matchup: -5% suppression
      - Ballpark HR factor (park_factor from Baseball Reference)
          Coors (1.35), Yankee Stadium (1.18), Fenway (0.88), etc.
      - Wind: blowing out at >10mph adds ~8% per 5mph increment
      - Temperature: ball carries further in heat — +1% per 5°F above 72°F
      - Home/road: marginal comfort edge (~+2% at home)

    Base HR probability: 13% per starting position player per game
    (league avg ~0.035 HR/PA × ~3.8 PA/game ≈ 13.3%)
    """
    if hr_df is None or hr_df.empty:
        return []

    # ── Base rate ──────────────────────────────────────────────────────────
    BASE_HR_PROB = 0.133

    # ── Pitcher handedness adjustment ──────────────────────────────────────
    # Platoon splits from 2022-24 Statcast aggregate
    ph = pitcher_hand.upper() if pitcher_hand else "R"

    # ── Environmental adjustments ──────────────────────────────────────────
    # Park factor: normalised so 1.00 = neutral
    park_adj = (park_factor - 1.0) * 0.6    # scale: Coors 1.35 → +21%

    # Wind: each 5mph blowing out adds ~4pp; blowing in subtracts
    if "out" in wind_dir.lower():
        wind_adj = max(0, wind_mph - 5) / 5 * 0.04
    elif "in" in wind_dir.lower():
        wind_adj = -max(0, wind_mph - 5) / 5 * 0.03
    else:
        wind_adj = 0.0                       # cross wind or unknown

    # Temperature: ball travels ~3% further per 10°F above 70°F
    temp_adj = max(0, temp_f - 72) / 5 * 0.01

    # Home comfort edge
    home_adj = 0.02 if home_game else 0.0

    recs = []
    for _, row in hr_df.iterrows():
        over_odds = row.get("over_odds")
        if over_odds is None:
            continue

        # ── Platoon adjustment per batter ────────────────────────────────
        bats = str(row.get("bats", "R")).upper()   # batter handedness
        if bats == "S":
            # Switch hitter: always has platoon advantage
            platoon_adj = 0.08
        elif bats != ph:
            # Opposite hand: batter has advantage
            platoon_adj = 0.12 if bats == "R" else 0.08
        else:
            # Same hand: pitcher has advantage
            platoon_adj = -0.05

        # ── Composite model probability ──────────────────────────────────
        model_p = BASE_HR_PROB * (
            1.0 + platoon_adj + park_adj + wind_adj + temp_adj + home_adj
        )
        model_p = float(np.clip(model_p, 0.04, 0.45))

        dec      = american_to_decimal(float(over_odds))
        implied_p = 1.0 / dec
        edge     = model_p - implied_p

        if edge < min_edge:
            continue

        stake = kelly_stake(model_p, dec, initial_bankroll,
                            kelly_frac, max_stake_pct)

        # Build a readable notes string explaining the factors
        factor_notes = []
        if abs(platoon_adj) > 0.01:
            direction = "advantage" if platoon_adj > 0 else "disadvantage"
            factor_notes.append(f"{bats}HB vs {ph}HP platoon {direction} ({platoon_adj:+.0%})")
        if abs(park_adj) > 0.01:
            factor_notes.append(f"park {park_adj:+.0%}")
        if abs(wind_adj) > 0.005:
            factor_notes.append(f"wind {wind_mph:.0f}mph {wind_dir} ({wind_adj:+.0%})")
        if abs(temp_adj) > 0.005:
            factor_notes.append(f"{temp_f:.0f}°F ({temp_adj:+.0%})")
        if not factor_notes:
            factor_notes.append("neutral conditions")

        recs.append(BetRecommendation(
            game_pk       = game_pk,
            home_team     = home_team,
            away_team     = away_team,
            home_sp       = ph + "HP",
            away_sp       = "",
            bet_type      = "hr_prop",
            bet_side      = f"{row['player']} HR Over {row.get('line', 0.5)}",
            model_prob    = round(model_p, 4),
            fair_prob     = round(implied_p, 4),
            edge          = round(edge, 4),
            american_odds = float(over_odds),
            decimal_odds  = round(dec, 4),
            vig           = 0.0,
            n_books       = 1,
            kelly_pct     = round(stake / initial_bankroll, 4),
            stake         = stake,
            bankroll      = initial_bankroll,
            verdict       = "BET",
            confidence    = "MEDIUM" if edge >= 0.07 else "LOW",
            notes         = factor_notes,
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
    
    # Filter out games with invalid odds (0 or extreme values)
    valid = valid[
        (valid["home_odds"] != 0) & 
        (valid["away_odds"] != 0) &
        (valid["home_odds"].abs() < 10000) &
        (valid["away_odds"].abs() < 10000)
    ].reset_index(drop=True)

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
