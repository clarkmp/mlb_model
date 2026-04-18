"""
mlb_features.py
---------------
Builds the full MLB feature matrix from raw game + stats data.

Feature groups:
  1. Recent team form      — last 10 / 30 game rolling win%, run differential
  2. Pythagorean win pct   — true-talent indicator from runs scored/allowed
  3. Park factors          — run environment per ballpark
  4. Rest / travel         — days since last game
  5. Head-to-head          — recent H2H win rate for this exact matchup
  6. Season team stats     — batting OPS, ERA, WHIP, FIP (when API available)
  7. Derived differentials — home minus away for each stat group

All rolling stats use shift(1) before the window — no lookahead leakage.
Doubleheaders are handled by dropping duplicate date+team keys (keep last).
Season stats columns are treated as optional; the model degrades gracefully
to core features if the MLB Stats API is unreachable.
"""

import numpy as np
import pandas as pd
from typing import Optional
from mlb_data import get_park_factor, compute_h2h


# ─────────────────────────────────────────────
# Rolling team stats (vectorised, no apply)
# ─────────────────────────────────────────────

def _rolling_team_stats(df: pd.DataFrame, window: int,
                        col_suffix: str) -> pd.DataFrame:
    """
    Compute rolling stats for every team from their own game log,
    then map the results back onto each game row (home + away sides).
    Uses shift(1) to prevent any lookahead.
    Handles doubleheaders by keeping only the last entry per date+team.
    """
    df = df.sort_values("game_date").reset_index(drop=True)

    home_log = df[["game_date", "home_team",
                   "home_score", "away_score", "home_win"]].copy()
    home_log.columns = ["date", "team", "runs_scored", "runs_allowed", "win"]

    away_log = df[["game_date", "away_team",
                   "away_score", "home_score", "home_win"]].copy()
    away_log["win"] = 1 - away_log["home_win"]
    away_log = away_log.drop(columns="home_win")
    away_log.columns = ["date", "team", "runs_scored", "runs_allowed", "win"]

    log = (pd.concat([home_log, away_log])
             .sort_values("date")
             .reset_index(drop=True))

    min_p = max(3, window // 3)
    for stat in ["runs_scored", "runs_allowed", "win"]:
        log[f"roll_{stat}"] = (
            log.groupby("team")[stat]
               .transform(lambda x: x.shift(1).rolling(window, min_periods=min_p).mean())
        )

    log["roll_run_diff"] = log["roll_runs_scored"] - log["roll_runs_allowed"]

    # Build lookup dict — deduplicate doubleheaders (keep last game of the day)
    log["_key"] = (pd.to_datetime(log["date"]).dt.strftime("%Y-%m-%d")
                   + "__" + log["team"])
    log = log.drop_duplicates(subset="_key", keep="last")

    roll_cols = ["roll_runs_scored", "roll_runs_allowed",
                 "roll_win", "roll_run_diff"]
    lookup = {col: log.set_index("_key")[col].to_dict() for col in roll_cols}

    def pull(date_val, team, col):
        key = pd.Timestamp(date_val).strftime("%Y-%m-%d") + "__" + team
        val = lookup[col].get(key)
        return float(val) if (val is not None and val == val) else np.nan

    s = col_suffix
    df[f"home_win_pct_{s}"]  = [pull(r.game_date, r.home_team, "roll_win")          for _, r in df.iterrows()]
    df[f"away_win_pct_{s}"]  = [pull(r.game_date, r.away_team, "roll_win")          for _, r in df.iterrows()]
    df[f"home_rs_{s}"]       = [pull(r.game_date, r.home_team, "roll_runs_scored")  for _, r in df.iterrows()]
    df[f"away_rs_{s}"]       = [pull(r.game_date, r.away_team, "roll_runs_scored")  for _, r in df.iterrows()]
    df[f"home_ra_{s}"]       = [pull(r.game_date, r.home_team, "roll_runs_allowed") for _, r in df.iterrows()]
    df[f"away_ra_{s}"]       = [pull(r.game_date, r.away_team, "roll_runs_allowed") for _, r in df.iterrows()]
    df[f"home_run_diff_{s}"] = [pull(r.game_date, r.home_team, "roll_run_diff")     for _, r in df.iterrows()]
    df[f"away_run_diff_{s}"] = [pull(r.game_date, r.away_team, "roll_run_diff")     for _, r in df.iterrows()]
    df[f"win_pct_diff_{s}"]  = df[f"home_win_pct_{s}"]  - df[f"away_win_pct_{s}"]
    df[f"run_diff_diff_{s}"] = df[f"home_run_diff_{s}"] - df[f"away_run_diff_{s}"]

    return df


# ─────────────────────────────────────────────
# Rest days
# ─────────────────────────────────────────────

def _add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True)
    last_game: dict = {}
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        gd = pd.Timestamp(row["game_date"])
        for team, rest_list in [(row["home_team"], home_rest),
                                (row["away_team"], away_rest)]:
            last = last_game.get(team)
            rest_list.append(min(int((gd - last).days), 7) if last else 3)
            last_game[team] = gd

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    return df


# ─────────────────────────────────────────────
# Pythagorean expectation
# ─────────────────────────────────────────────

def _pythagorean(rs: float, ra: float, exp: float = 1.83) -> float:
    if rs <= 0 or ra <= 0:
        return 0.5
    return rs ** exp / (rs ** exp + ra ** exp)


def _add_pythagorean(df: pd.DataFrame) -> pd.DataFrame:
    df["home_pyth_win_pct"] = [
        _pythagorean(r.get("home_rs_L10", 4.5), r.get("home_ra_L10", 4.5))
        for _, r in df.iterrows()
    ]
    df["away_pyth_win_pct"] = [
        _pythagorean(r.get("away_rs_L10", 4.5), r.get("away_ra_L10", 4.5))
        for _, r in df.iterrows()
    ]
    df["pyth_diff"] = df["home_pyth_win_pct"] - df["away_pyth_win_pct"]
    return df


# ─────────────────────────────────────────────
# Park factor
# ─────────────────────────────────────────────

def _add_park_factor(df: pd.DataFrame) -> pd.DataFrame:
    df["park_factor"] = df["home_team"].apply(get_park_factor)
    return df


# ─────────────────────────────────────────────
# Head-to-head
# ─────────────────────────────────────────────

def _add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    h2h = compute_h2h(df, window_games=20)
    df["h2h_home_win_rate"] = df.apply(
        lambda r: h2h.get((r["home_team"], r["away_team"]), 0.5), axis=1
    )
    return df


# ─────────────────────────────────────────────
# Season stats merge  (safe — never crashes on missing columns)
# ─────────────────────────────────────────────

# These are the stat columns we want from the team stats API.
# Any that are missing from the API response are silently skipped.
_WANTED_BAT = ["bat_avg", "bat_obp", "bat_slg", "bat_ops",
               "bat_homeRuns", "bat_strikeOuts", "bat_walks"]
_WANTED_PIT = ["pit_era", "pit_whip",
               "pit_strikeoutsPer9Inn", "pit_walksPer9Inn",
               "pit_homeRunsPer9", "pit_fieldingIndependent"]
_ALL_WANTED  = _WANTED_BAT + _WANTED_PIT


def merge_season_stats(games_df: pd.DataFrame,
                       team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join season stats onto games. Only joins columns that actually
    exist in team_stats_df — never crashes on missing API fields.
    """
    if team_stats_df is None or team_stats_df.empty:
        return games_df

    available = [c for c in _ALL_WANTED if c in team_stats_df.columns]
    if not available:
        return games_df

    ts = team_stats_df[["team_id", "season"] + available].copy()
    games_df = games_df.copy()
    games_df["_season"] = pd.to_datetime(games_df["game_date"]).dt.year

    for side, id_col in [("home", "home_team_id"), ("away", "away_team_id")]:
        if id_col not in games_df.columns:
            continue
        renamed = ts.rename(columns={c: f"{side}_{c}" for c in available})
        merged  = games_df.merge(
            renamed,
            left_on  = [id_col, "_season"],
            right_on = ["team_id", "season"],
            how      = "left",
            suffixes = ("", "_drop"),
        )
        # Drop duplicate key columns introduced by the merge
        merged = merged[[c for c in merged.columns if not c.endswith("_drop")]]
        for col in [f"{side}_{c}" for c in available]:
            if col in merged.columns:
                games_df[col] = merged[col].values

    games_df = games_df.drop(columns=["_season"], errors="ignore")
    return games_df


# ─────────────────────────────────────────────
# Feature column lists
# ─────────────────────────────────────────────

# Always available — computed from the game log alone
FEATURE_COLS_CORE = [
    "home_win_pct_L10", "away_win_pct_L10",
    "home_win_pct_L30", "away_win_pct_L30",
    "win_pct_diff_L10", "win_pct_diff_L30",
    "home_run_diff_L10", "away_run_diff_L10",
    "run_diff_diff_L10",
    "home_pyth_win_pct", "away_pyth_win_pct", "pyth_diff",
    "park_factor",
    "rest_advantage", "home_rest_days", "away_rest_days",
    "h2h_home_win_rate",
]

# Only active when the MLB Stats API returns season stats
FEATURE_COLS_EXTENDED = [
    "home_bat_ops",  "away_bat_ops",
    "home_bat_avg",  "away_bat_avg",
    "home_bat_obp",  "away_bat_obp",
    "home_pit_era",  "away_pit_era",
    "home_pit_whip", "away_pit_whip",
    "home_pit_fieldingIndependent", "away_pit_fieldingIndependent",
    "home_pit_strikeoutsPer9Inn",   "away_pit_strikeoutsPer9Inn",
    "ops_diff", "era_diff", "fip_diff",
]

# Active set — updated by build_features() at runtime
FEATURE_COLS: list = list(FEATURE_COLS_CORE)

TARGET_COL = "home_win"


# ─────────────────────────────────────────────
# Master feature builder
# ─────────────────────────────────────────────

def _safe_diff(df, col_a, col_b, fill_a=None, fill_b=None):
    """Compute col_a - col_b only if both columns exist and have data."""
    a = df.get(col_a, pd.Series(np.nan, index=df.index))
    b = df.get(col_b, pd.Series(np.nan, index=df.index))
    if fill_a is not None:
        a = a.fillna(fill_a)
    if fill_b is not None:
        b = b.fillna(fill_b)
    return a - b


def build_features(games_df: pd.DataFrame,
                   team_stats_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full feature pipeline. Returns a dataframe with FEATURE_COLS columns
    ready for modelling.

    team_stats_df is optional. When provided (i.e. the MLB Stats API is
    reachable), the extended feature set is activated. When absent the
    model falls back to core rolling / park / rest / H2H features.
    """
    global FEATURE_COLS

    df = games_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    print("  Building rolling form features (L10)...")
    df = _rolling_team_stats(df, window=10, col_suffix="L10")

    print("  Building rolling form features (L30)...")
    df = _rolling_team_stats(df, window=30, col_suffix="L30")

    print("  Adding rest features...")
    df = _add_rest_features(df)

    print("  Adding Pythagorean expectation...")
    df = _add_pythagorean(df)

    print("  Adding park factors...")
    df = _add_park_factor(df)

    print("  Adding head-to-head history...")
    df = _add_h2h(df)

    # ── Season stats (optional) ──────────────────────────────────────────
    has_season_stats = (
        team_stats_df is not None
        and not team_stats_df.empty
        and len([c for c in _ALL_WANTED if c in team_stats_df.columns]) > 0
    )

    if has_season_stats:
        print("  Merging season stats (batting + pitching)...")
        df = merge_season_stats(df, team_stats_df)

        # Safe differentials — only computed when both columns exist
        df["ops_diff"] = _safe_diff(df, "home_bat_ops", "away_bat_ops", 0.720, 0.720)
        df["era_diff"] = _safe_diff(df, "away_pit_era", "home_pit_era", 4.50, 4.50)
        df["fip_diff"] = _safe_diff(df,
            "away_pit_fieldingIndependent",
            "home_pit_fieldingIndependent", 4.50, 4.50)

        # Only include extended cols that were actually populated
        populated_extended = [
            c for c in FEATURE_COLS_EXTENDED
            if c in df.columns and df[c].notna().sum() > len(df) * 0.5
        ]
        FEATURE_COLS = list(FEATURE_COLS_CORE) + populated_extended
        print(f"  Extended features active: {len(populated_extended)} columns "
              f"({len(FEATURE_COLS)} total).")
    else:
        FEATURE_COLS = list(FEATURE_COLS_CORE)
        print(f"  Core features only ({len(FEATURE_COLS)} features). "
              f"Season stats unavailable — model uses form/park/rest/H2H.")

    # Ensure every active feature column exists (fill missing with NaN)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    n_valid = df.dropna(subset=FEATURE_COLS).shape[0]
    print(f"  {n_valid} / {len(df)} games have complete feature vectors")
    return df


# ─────────────────────────────────────────────
# Feature vector for a single upcoming game
# ─────────────────────────────────────────────

def build_game_features(
    home_team:        str,
    away_team:        str,
    history_df:       pd.DataFrame,
    home_sp_stats:    dict = None,
    away_sp_stats:    dict = None,
    home_lineup_stats: dict = None,
    away_lineup_stats: dict = None,
) -> dict:
    """
    Build a feature vector for one upcoming game from recent history
    plus optional live pitcher / lineup data fetched on the day.
    """
    active_cols    = list(FEATURE_COLS)
    league_medians = {
        col: float(history_df[col].median())
        for col in active_cols
        if col in history_df.columns and history_df[col].notna().any()
    }

    def last_row(team):
        mask = ((history_df["home_team"] == team) |
                (history_df["away_team"] == team))
        rows = history_df[mask].dropna(subset=["home_win_pct_L10"], how="all")
        return rows.iloc[-1] if not rows.empty else None

    home_row = last_row(home_team)
    away_row = last_row(away_team)
    features: dict = {}

    def extract_side(row, team, target_side):
        if row is None:
            return
        actual_side = "home" if row["home_team"] == team else "away"
        col_map = {
            f"{target_side}_win_pct_L10":  f"{actual_side}_win_pct_L10",
            f"{target_side}_win_pct_L30":  f"{actual_side}_win_pct_L30",
            f"{target_side}_run_diff_L10": f"{actual_side}_run_diff_L10",
            f"{target_side}_rs_L10":       f"{actual_side}_rs_L10",
            f"{target_side}_ra_L10":       f"{actual_side}_ra_L10",
            f"{target_side}_pyth_win_pct": f"{actual_side}_pyth_win_pct",
            f"{target_side}_bat_ops":      f"{actual_side}_bat_ops",
            f"{target_side}_bat_avg":      f"{actual_side}_bat_avg",
            f"{target_side}_bat_obp":      f"{actual_side}_bat_obp",
            f"{target_side}_pit_era":      f"{actual_side}_pit_era",
            f"{target_side}_pit_whip":     f"{actual_side}_pit_whip",
            f"{target_side}_pit_fieldingIndependent":
                f"{actual_side}_pit_fieldingIndependent",
            f"{target_side}_pit_strikeoutsPer9Inn":
                f"{actual_side}_pit_strikeoutsPer9Inn",
            f"{target_side}_rest_days":    f"{actual_side}_rest_days",
        }
        for tgt, src in col_map.items():
            val = row.get(src, np.nan)
            features[tgt] = (float(val)
                             if val is not None and pd.notna(val)
                             else league_medians.get(tgt, np.nan))

    extract_side(home_row, home_team, "home")
    extract_side(away_row, away_team, "away")

    # Differentials
    features["win_pct_diff_L10"]  = (features.get("home_win_pct_L10",  0.5)
                                   - features.get("away_win_pct_L10",  0.5))
    features["win_pct_diff_L30"]  = (features.get("home_win_pct_L30",  0.5)
                                   - features.get("away_win_pct_L30",  0.5))
    features["run_diff_diff_L10"] = (features.get("home_run_diff_L10", 0.0)
                                   - features.get("away_run_diff_L10", 0.0))
    features["pyth_diff"]         = (features.get("home_pyth_win_pct", 0.5)
                                   - features.get("away_pyth_win_pct", 0.5))

    features["park_factor"]     = get_park_factor(home_team)
    features["home_rest_days"]  = features.get("home_rest_days", 2)
    features["away_rest_days"]  = features.get("away_rest_days", 2)
    features["rest_advantage"]  = (features["home_rest_days"]
                                 - features["away_rest_days"])

    h2h = compute_h2h(history_df)
    features["h2h_home_win_rate"] = h2h.get((home_team, away_team), 0.5)

    # Override with live SP stats if available
    for side, sp_stats in [("home", home_sp_stats), ("away", away_sp_stats)]:
        if sp_stats:
            for dest, src, default in [
                (f"{side}_pit_era",  "era",   4.50),
                (f"{side}_pit_whip", "whip",  1.30),
                (f"{side}_pit_fieldingIndependent", "fip",   4.50),
                (f"{side}_pit_strikeoutsPer9Inn",   "k_per_9", 8.0),
            ]:
                if dest in active_cols:
                    features[dest] = sp_stats.get(src, features.get(dest, default))

    # Override with live lineup stats if available
    for side, lu_stats in [("home", home_lineup_stats), ("away", away_lineup_stats)]:
        if lu_stats:
            for dest, src, default in [
                (f"{side}_bat_ops", "lineup_ops", 0.720),
                (f"{side}_bat_avg", "lineup_avg", 0.250),
                (f"{side}_bat_obp", "lineup_obp", 0.320),
            ]:
                if dest in active_cols:
                    features[dest] = lu_stats.get(src, features.get(dest, default))

    # Recalculate differentials after live overrides
    features["ops_diff"] = (features.get("home_bat_ops", 0.720)
                          - features.get("away_bat_ops", 0.720))
    features["era_diff"] = (features.get("away_pit_era", 4.50)
                          - features.get("home_pit_era", 4.50))
    features["fip_diff"] = (features.get("away_pit_fieldingIndependent", 4.50)
                          - features.get("home_pit_fieldingIndependent", 4.50))

    # Fill any remaining gaps with league medians
    for col in active_cols:
        if col not in features or (
            isinstance(features[col], float) and np.isnan(features[col])
        ):
            features[col] = league_medians.get(col, 0.0)

    return features
