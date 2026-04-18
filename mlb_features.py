"""
mlb_features.py
---------------
Builds the full MLB feature matrix from raw game + stats data.

Feature groups:
  1. Recent team form          — last 10/30 game rolling win%, run differential
  2. Starting pitcher quality  — ERA, WHIP, FIP, K/9, recent form
  3. Lineup strength           — OPS, contact rate, power
  4. Bullpen                   — team ERA in relief situations
  5. Park factors              — run environment adjustment
  6. Rest / travel             — days off, home/away streak
  7. Head-to-head              — recent matchup history
  8. Run line context          — avg run differential (helps calibrate spread)
  9. Pythagorean expectation   — true talent indicator from runs scored/allowed
"""

import numpy as np
import pandas as pd
from typing import Optional

from mlb_data import get_park_factor, compute_h2h


# ─────────────────────────────────────────────
# Rolling team stats
# ─────────────────────────────────────────────

def _rolling_team_stats(df: pd.DataFrame, window: int, col_suffix: str) -> pd.DataFrame:
    """
    For each game, compute rolling stats for home and away teams
    using only games BEFORE the current game (shift(1) prevents lookahead).
    """
    df = df.sort_values("game_date").reset_index(drop=True)

    # Build per-team game log (each game appears twice: once as home, once as away)
    home_log = df[["game_date", "home_team", "home_score", "away_score", "home_win"]].copy()
    home_log.columns = ["date", "team", "runs_scored", "runs_allowed", "win"]

    away_log = df[["game_date", "away_team", "away_score", "home_score", "home_win"]].copy()
    away_log["home_win"] = 1 - away_log["home_win"]
    away_log.columns = ["date", "team", "runs_scored", "runs_allowed", "win"]

    log = pd.concat([home_log, away_log]).sort_values("date").reset_index(drop=True)

    for stat in ["runs_scored", "runs_allowed", "win"]:
        log[f"roll_{stat}_{window}"] = (
            log.groupby("team")[stat]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=max(3, window//3)).mean())
        )

    log["roll_run_diff"] = log[f"roll_runs_scored_{window}"] - log[f"roll_runs_allowed_{window}"]
    log["_key"] = pd.to_datetime(log["date"]).dt.strftime("%Y-%m-%d") + "__" + log["team"]

    # Doubleheaders: same team plays twice on same date → duplicate keys.
    # Keep the LAST entry so the rolling stat includes all prior games.
    log = log.drop_duplicates(subset="_key", keep="last")

    stat_cols = [f"roll_runs_scored_{window}", f"roll_runs_allowed_{window}",
                 f"roll_win_{window}", "roll_run_diff"]
    stats_dict = {col: log.set_index("_key")[col].to_dict() for col in stat_cols}

    def pull(date, team, col):
        key = pd.Timestamp(date).strftime("%Y-%m-%d") + "__" + team
        val = stats_dict[col].get(key, np.nan)
        return float(val) if (val is not None and val == val) else np.nan

    suffix = col_suffix
    df[f"home_win_pct_{suffix}"]    = df.apply(lambda r: pull(r.game_date, r.home_team, f"roll_win_{window}"), axis=1)
    df[f"away_win_pct_{suffix}"]    = df.apply(lambda r: pull(r.game_date, r.away_team, f"roll_win_{window}"), axis=1)
    df[f"home_rs_{suffix}"]         = df.apply(lambda r: pull(r.game_date, r.home_team, f"roll_runs_scored_{window}"), axis=1)
    df[f"away_rs_{suffix}"]         = df.apply(lambda r: pull(r.game_date, r.away_team, f"roll_runs_scored_{window}"), axis=1)
    df[f"home_ra_{suffix}"]         = df.apply(lambda r: pull(r.game_date, r.home_team, f"roll_runs_allowed_{window}"), axis=1)
    df[f"away_ra_{suffix}"]         = df.apply(lambda r: pull(r.game_date, r.away_team, f"roll_runs_allowed_{window}"), axis=1)
    df[f"home_run_diff_{suffix}"]   = df.apply(lambda r: pull(r.game_date, r.home_team, "roll_run_diff"), axis=1)
    df[f"away_run_diff_{suffix}"]   = df.apply(lambda r: pull(r.game_date, r.away_team, "roll_run_diff"), axis=1)
    df[f"win_pct_diff_{suffix}"]    = df[f"home_win_pct_{suffix}"] - df[f"away_win_pct_{suffix}"]
    df[f"run_diff_diff_{suffix}"]   = df[f"home_run_diff_{suffix}"] - df[f"away_run_diff_{suffix}"]

    return df


# ─────────────────────────────────────────────
# Rest / home-field streaks
# ─────────────────────────────────────────────

def _add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True)

    home_dates = {}
    away_dates = {}
    home_rest_list, away_rest_list = [], []
    home_streak_list, away_streak_list = [], []

    for _, row in df.iterrows():
        gd = pd.Timestamp(row["game_date"])

        # Rest days
        for team_col, dates_dict, rest_list in [
            ("home_team", home_dates, home_rest_list),
            ("away_team", away_dates, away_rest_list),
        ]:
            team = row[team_col]
            last = dates_dict.get(team)
            rest = (gd - last).days if last else 3
            rest_list.append(min(rest, 7))
            dates_dict[team] = gd

    df["home_rest_days"] = home_rest_list
    df["away_rest_days"] = away_rest_list
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    return df


# ─────────────────────────────────────────────
# Pythagorean expectation (true talent win%)
# ─────────────────────────────────────────────

def _pythagorean(rs: float, ra: float, exp: float = 1.83) -> float:
    """Baseball Pythagorean theorem: expected win % from runs scored/allowed."""
    if rs <= 0 or ra <= 0:
        return 0.5
    return rs ** exp / (rs ** exp + ra ** exp)


def _add_pythagorean(df: pd.DataFrame) -> pd.DataFrame:
    for side, rs_col, ra_col in [
        ("home", "home_rs_L10", "home_ra_L10"),
        ("away", "away_rs_L10", "away_ra_L10"),
    ]:
        df[f"{side}_pyth_win_pct"] = df.apply(
            lambda r: _pythagorean(r.get(rs_col, 4.5), r.get(ra_col, 4.5)), axis=1
        )
    df["pyth_diff"] = df["home_pyth_win_pct"] - df["away_pyth_win_pct"]
    return df


# ─────────────────────────────────────────────
# Pitcher features
# ─────────────────────────────────────────────

def _pitcher_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute rolling ERA and WHIP from the last N starts
    recorded in the game log (using the sp columns if present).
    This is a proxy for pitcher form when we don't have per-pitcher IDs.
    """
    # We use team-level pitching proxy: runs allowed per game rolling
    # More granular pitcher stats are in mlb_data.fetch_pitcher_stats
    # and get attached in build_features_for_game at predict time.
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
# Season-level team stats merge
# ─────────────────────────────────────────────

def merge_season_stats(games_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join season-level batting/pitching stats onto the game dataframe.
    Selects key rate stats and renames to home_/away_ prefixes.
    """
    if team_stats_df.empty:
        return games_df

    # Key columns to include
    stat_cols = [
        "bat_avg", "bat_obp", "bat_slg", "bat_ops",
        "bat_homeRuns", "bat_strikeOuts", "bat_walks",
        "pit_era", "pit_whip", "pit_strikeoutsPer9Inn",
        "pit_walksPer9Inn", "pit_homeRunsPer9",
        "pit_fieldingIndependent",
    ]
    available = [c for c in stat_cols if c in team_stats_df.columns]

    # Merge with safe fall-through on missing cols
    ts = team_stats_df[["team_id", "season"] + available].copy()

    games_df = games_df.copy()
    games_df["_season"] = pd.to_datetime(games_df["game_date"]).dt.year

    for side, id_col in [("home", "home_team_id"), ("away", "away_team_id")]:
        merged = games_df.merge(
            ts.rename(columns={c: f"{side}_{c}" for c in available}),
            left_on=[id_col, "_season"],
            right_on=["team_id", "season"],
            how="left",
        )
        for col in [f"{side}_{c}" for c in available]:
            games_df[col] = merged[col].values

    return games_df


# ─────────────────────────────────────────────
# Master feature builder
# ─────────────────────────────────────────────

# Core features — always available from game log alone
FEATURE_COLS_CORE = [
    "home_win_pct_L10", "away_win_pct_L10",
    "home_win_pct_L30", "away_win_pct_L30",
    "win_pct_diff_L10", "win_pct_diff_L30",
    "home_run_diff_L10", "away_run_diff_L10",
    "run_diff_diff_L10",
    "home_pyth_win_pct", "away_pyth_win_pct", "pyth_diff",
    "park_factor", "rest_advantage",
    "home_rest_days", "away_rest_days",
    "h2h_home_win_rate",
]

# Extended features — populated when team stats API is reachable
FEATURE_COLS_EXTENDED = [
    "home_bat_ops", "away_bat_ops",
    "home_bat_avg", "away_bat_avg",
    "home_bat_obp", "away_bat_obp",
    "home_pit_era", "away_pit_era",
    "home_pit_whip", "away_pit_whip",
    "home_pit_fieldingIndependent", "away_pit_fieldingIndependent",
    "home_pit_strikeoutsPer9Inn", "away_pit_strikeoutsPer9Inn",
    "ops_diff", "era_diff", "fip_diff",
]

# Active feature set — starts as core, expands if extended data is available
FEATURE_COLS = list(FEATURE_COLS_CORE)

TARGET_COL = "home_win"


def build_features(games_df: pd.DataFrame,
                   team_stats_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full feature pipeline. Takes raw game records and returns
    a dataframe with FEATURE_COLS ready for modelling.

    FEATURE_COLS is dynamically updated: starts as core rolling/park/rest
    features and expands to include season batting/pitching stats when
    team_stats_df is provided (requires live MLB Stats API access).
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

    has_season_stats = team_stats_df is not None and not team_stats_df.empty
    if has_season_stats:
        print("  Merging season stats (batting + pitching)...")
        df = merge_season_stats(df, team_stats_df)

        # Derived differentials
        df["ops_diff"] = df["home_bat_ops"] - df["away_bat_ops"]
        df["era_diff"] = df["away_pit_era"] - df["home_pit_era"]
        df["fip_diff"] = df["away_pit_fieldingIndependent"] - df["home_pit_fieldingIndependent"]

        # Activate extended feature set
        FEATURE_COLS = list(FEATURE_COLS_CORE) + list(FEATURE_COLS_EXTENDED)
        print(f"  Extended features active ({len(FEATURE_COLS)} total).")
    else:
        FEATURE_COLS = list(FEATURE_COLS_CORE)
        print(f"  Core features only ({len(FEATURE_COLS)} features). "
              f"Season stats unavailable — model still runs on form/park/rest/H2H.")

    # Ensure all active feature columns exist
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
    home_team: str,
    away_team: str,
    history_df: pd.DataFrame,
    home_sp_stats: dict = None,
    away_sp_stats: dict = None,
    home_lineup_stats: dict = None,
    away_lineup_stats: dict = None,
) -> dict:
    """
    Build a feature vector for a single upcoming game using
    recent history + optional live pitcher/lineup data.
    """
    league_medians = {col: history_df[col].median()
                      for col in FEATURE_COLS if col in history_df.columns}

    # Pull the most recent row where this team played
    def last_row(team):
        mask = (history_df["home_team"] == team) | (history_df["away_team"] == team)
        rows = history_df[mask].dropna(subset=["home_win_pct_L10"])
        return rows.iloc[-1] if not rows.empty else None

    home_row = last_row(home_team)
    away_row = last_row(away_team)

    features = {}

    def get_side_features(row, team, target_side):
        if row is None:
            return
        actual_side = "home" if row["home_team"] == team else "away"
        swap = actual_side != target_side

        col_pairs = [
            (f"{target_side}_win_pct_L10",  f"{actual_side}_win_pct_L10"),
            (f"{target_side}_win_pct_L30",  f"{actual_side}_win_pct_L30"),
            (f"{target_side}_run_diff_L10", f"{actual_side}_run_diff_L10"),
            (f"{target_side}_rs_L10",       f"{actual_side}_rs_L10"),
            (f"{target_side}_ra_L10",       f"{actual_side}_ra_L10"),
            (f"{target_side}_pyth_win_pct", f"{actual_side}_pyth_win_pct"),
            (f"{target_side}_bat_ops",      f"{actual_side}_bat_ops"),
            (f"{target_side}_bat_avg",      f"{actual_side}_bat_avg"),
            (f"{target_side}_bat_obp",      f"{actual_side}_bat_obp"),
            (f"{target_side}_pit_era",      f"{actual_side}_pit_era"),
            (f"{target_side}_pit_whip",     f"{actual_side}_pit_whip"),
            (f"{target_side}_pit_fieldingIndependent", f"{actual_side}_pit_fieldingIndependent"),
            (f"{target_side}_pit_strikeoutsPer9Inn",   f"{actual_side}_pit_strikeoutsPer9Inn"),
        ]
        for target_col, source_col in col_pairs:
            val = row.get(source_col, np.nan)
            features[target_col] = float(val) if pd.notna(val) else league_medians.get(target_col, np.nan)

    get_side_features(home_row, home_team, "home")
    get_side_features(away_row, away_team, "away")

    # Differentials
    features["win_pct_diff_L10"] = features.get("home_win_pct_L10", 0.5) - features.get("away_win_pct_L10", 0.5)
    features["win_pct_diff_L30"] = features.get("home_win_pct_L30", 0.5) - features.get("away_win_pct_L30", 0.5)
    features["run_diff_diff_L10"] = features.get("home_run_diff_L10", 0.0) - features.get("away_run_diff_L10", 0.0)
    features["ops_diff"] = features.get("home_bat_ops", 0.720) - features.get("away_bat_ops", 0.720)
    features["era_diff"] = features.get("away_pit_era", 4.50) - features.get("home_pit_era", 4.50)
    features["fip_diff"] = features.get("away_pit_fieldingIndependent", 4.50) - features.get("home_pit_fieldingIndependent", 4.50)
    features["pyth_diff"] = features.get("home_pyth_win_pct", 0.5) - features.get("away_pyth_win_pct", 0.5)

    # Park factor
    features["park_factor"] = get_park_factor(home_team)

    # Rest (default 2 days if unknown)
    features["home_rest_days"] = 2
    features["away_rest_days"] = 2
    features["rest_advantage"] = 0

    # Head-to-head
    h2h = compute_h2h(history_df)
    features["h2h_home_win_rate"] = h2h.get((home_team, away_team), 0.5)

    # Override ERA/WHIP with actual starting pitcher stats if available
    if home_sp_stats:
        features["home_pit_era"]  = home_sp_stats.get("era",  features.get("home_pit_era", 4.50))
        features["home_pit_whip"] = home_sp_stats.get("whip", features.get("home_pit_whip", 1.30))
        features["home_pit_fieldingIndependent"] = home_sp_stats.get("fip", features.get("home_pit_fieldingIndependent", 4.50))
        features["home_pit_strikeoutsPer9Inn"]   = home_sp_stats.get("k_per_9", features.get("home_pit_strikeoutsPer9Inn", 8.0))

    if away_sp_stats:
        features["away_pit_era"]  = away_sp_stats.get("era",  features.get("away_pit_era", 4.50))
        features["away_pit_whip"] = away_sp_stats.get("whip", features.get("away_pit_whip", 1.30))
        features["away_pit_fieldingIndependent"] = away_sp_stats.get("fip", features.get("away_pit_fieldingIndependent", 4.50))
        features["away_pit_strikeoutsPer9Inn"]   = away_sp_stats.get("k_per_9", features.get("away_pit_strikeoutsPer9Inn", 8.0))

    # Recalculate differentials with live pitcher data
    features["era_diff"] = features.get("away_pit_era", 4.50) - features.get("home_pit_era", 4.50)
    features["fip_diff"] = features.get("away_pit_fieldingIndependent", 4.50) - features.get("home_pit_fieldingIndependent", 4.50)

    # Override lineup stats if available
    if home_lineup_stats:
        features["home_bat_ops"] = home_lineup_stats.get("lineup_ops", features.get("home_bat_ops", 0.720))
        features["home_bat_avg"] = home_lineup_stats.get("lineup_avg", features.get("home_bat_avg", 0.250))
        features["home_bat_obp"] = home_lineup_stats.get("lineup_obp", features.get("home_bat_obp", 0.320))

    if away_lineup_stats:
        features["away_bat_ops"] = away_lineup_stats.get("lineup_ops", features.get("away_bat_ops", 0.720))
        features["away_bat_avg"] = away_lineup_stats.get("lineup_avg", features.get("away_bat_avg", 0.250))
        features["away_bat_obp"] = away_lineup_stats.get("lineup_obp", features.get("away_bat_obp", 0.320))

    features["ops_diff"] = features.get("home_bat_ops", 0.720) - features.get("away_bat_ops", 0.720)

    # Fill any remaining NaNs with medians
    for col in FEATURE_COLS:
        if col not in features or (isinstance(features[col], float) and np.isnan(features[col])):
            features[col] = league_medians.get(col, 0.0)

    return features
