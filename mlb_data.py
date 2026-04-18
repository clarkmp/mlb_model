"""
mlb_data.py
-----------
All data fetching for the MLB model.

Sources:
  - MLB Stats API (statsapi.mlb.com) — free, no key needed
    * Game schedules and results
    * Starting pitchers
    * Team batting/pitching stats
    * Live game data
  - The Odds API — live moneyline odds (requires ODDS_API_KEY in .env)
  - Baseball Reference style stats via MLB API aggregation

Install: pip install requests python-dotenv
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_KEY = os.getenv("ODDS_API_KEY", "")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get(url: str, params: dict = None, cache_key: str = "",
         cache_ttl_mins: int = 60, base: str = "") -> dict | list:
    """GET with optional file cache."""
    if cache_key:
        cf = CACHE_DIR / f"{cache_key}.json"
        if cf.exists():
            age = (time.time() - cf.stat().st_mtime) / 60
            if age < cache_ttl_mins:
                with open(cf) as f:
                    return json.load(f)

    full_url = (base or "") + url
    resp = requests.get(full_url, params=params or {}, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if cache_key:
        with open(CACHE_DIR / f"{cache_key}.json", "w") as f:
            json.dump(data, f)
    return data


def _mlb(endpoint: str, params: dict = None, cache_key: str = "",
         cache_ttl_mins: int = 360) -> dict:
    return _get(f"{MLB_API}/{endpoint}", params, cache_key, cache_ttl_mins)


# ─────────────────────────────────────────────
# Schedule & Results
# ─────────────────────────────────────────────

def fetch_schedule(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch completed MLB games between two dates.
    Dates: 'YYYY-MM-DD'
    Returns one row per game with basic result info.
    """
    ck = f"schedule_{start_date}_{end_date}"
    data = _mlb("schedule", {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": "R",          # Regular season only
        "hydrate": "linescore,decisions,probablePitcher(note)",
        "fields": "dates,games,gamePk,gameDate,status,teams,score,"
                  "probablePitcher,decisions,linescore",
    }, cache_key=ck, cache_ttl_mins=720)

    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue

            home = g["teams"]["home"]
            away = g["teams"]["away"]
            ls = g.get("linescore", {})

            def sp_name(team_key):
                dec = g.get("decisions", {})
                # Try decisions first (actual starter)
                for role in ["winner", "loser"]:
                    p = dec.get(role, {})
                    if p:
                        pass  # decisions don't tell us which team
                # Fall back to probablePitcher per team
                return (
                    g["teams"][team_key]
                    .get("probablePitcher", {})
                    .get("fullName", "Unknown")
                )

            rows.append({
                "game_pk":          g["gamePk"],
                "game_date":        g["gameDate"][:10],
                "home_team":        home["team"]["name"],
                "away_team":        away["team"]["name"],
                "home_team_id":     home["team"]["id"],
                "away_team_id":     away["team"]["id"],
                "home_score":       home.get("score", 0),
                "away_score":       away.get("score", 0),
                "home_win":         int(home.get("isWinner", False)),
                "home_runs":        ls.get("teams", {}).get("home", {}).get("runs", 0),
                "away_runs":        ls.get("teams", {}).get("away", {}).get("runs", 0),
                "innings":          ls.get("currentInning", 9),
                "home_sp":          sp_name("home"),
                "away_sp":          sp_name("away"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  Fetched {len(df)} completed games ({start_date} → {end_date})")
    return df


def fetch_season_schedule(seasons: list[int] = None) -> pd.DataFrame:
    """Fetch full season(s) of completed games."""
    if seasons is None:
        seasons = [2023, 2024]

    all_dfs = []
    for season in seasons:
        print(f"  Fetching {season} season...")
        # MLB regular season: ~Apr 1 – Oct 1
        df = fetch_schedule(f"{season}-04-01", f"{season}-10-01")
        all_dfs.append(df)
        time.sleep(0.5)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(f"  Total games: {len(combined)}")
    return combined


# ─────────────────────────────────────────────
# Team Stats
# ─────────────────────────────────────────────

def fetch_team_season_stats(season: int) -> pd.DataFrame:
    """
    Fetch team-level batting and pitching stats for a full season.
    Returns one row per team with key rate stats.
    """
    ck = f"team_stats_{season}"
    # Batting
    bat = _mlb("teams/stats", {
        "stats": "season",
        "group": "hitting",
        "season": season,
        "sportId": 1,
    }, cache_key=f"team_bat_{season}", cache_ttl_mins=1440)

    # Pitching
    pit = _mlb("teams/stats", {
        "stats": "season",
        "group": "pitching",
        "season": season,
        "sportId": 1,
    }, cache_key=f"team_pit_{season}", cache_ttl_mins=1440)

    def parse_stats(data, prefix):
        rows = {}
        for entry in data.get("stats", [{}])[0].get("splits", []):
            team = entry.get("team", {})
            tid = team.get("id")
            name = team.get("name", "")
            s = entry.get("stat", {})
            rows[tid] = {"team_id": tid, "team_name": name}
            for k, v in s.items():
                try:
                    rows[tid][f"{prefix}_{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
        return rows

    bat_rows = parse_stats(bat, "bat")
    pit_rows = parse_stats(pit, "pit")

    all_ids = set(bat_rows) | set(pit_rows)
    merged = []
    for tid in all_ids:
        row = {**bat_rows.get(tid, {}), **pit_rows.get(tid, {})}
        row["season"] = season
        merged.append(row)

    df = pd.DataFrame(merged)
    print(f"  Fetched season stats for {len(df)} teams ({season})")
    return df


# ─────────────────────────────────────────────
# Starting Pitcher Stats
# ─────────────────────────────────────────────

def fetch_pitcher_stats(pitcher_id: int, season: int) -> dict:
    """Fetch season stats for a specific pitcher."""
    ck = f"pitcher_{pitcher_id}_{season}"
    data = _mlb(f"people/{pitcher_id}/stats", {
        "stats": "season",
        "group": "pitching",
        "season": season,
    }, cache_key=ck, cache_ttl_mins=1440)

    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return {}
    s = splits[0].get("stat", {})
    return {
        "era":        float(s.get("era", 4.50)),
        "whip":       float(s.get("whip", 1.30)),
        "k_per_9":    float(s.get("strikeoutsPer9Inn", 8.0)),
        "bb_per_9":   float(s.get("walksPer9Inn", 3.0)),
        "fip":        float(s.get("fieldingIndependent", 4.50)),
        "innings":    float(s.get("inningsPitched", 0)),
        "k_pct":      float(s.get("strikeoutWalkRatio", 2.0)),
        "hr_per_9":   float(s.get("homeRunsPer9", 1.0)),
        "opp_avg":    float(s.get("avg", ".250").replace(".", "0.") if "." in str(s.get("avg","")) else 0.250),
    }


def fetch_probable_pitchers(game_pk: int) -> dict:
    """
    Get probable pitchers for a specific game.
    Returns {'home': {id, name, ...}, 'away': {id, name, ...}}
    """
    ck = f"probables_{game_pk}"
    data = _mlb(f"game/{game_pk}/boxscore", cache_key=ck, cache_ttl_mins=120)

    result = {}
    for side in ["home", "away"]:
        team_data = data.get("teams", {}).get(side, {})
        pitchers = team_data.get("pitchers", [])
        sp_id = pitchers[0] if pitchers else None

        if sp_id:
            player = data.get("teams", {}).get(side, {}).get("players", {})
            pid_key = f"ID{sp_id}"
            p_info = player.get(pid_key, {}).get("person", {})
            result[side] = {
                "id":   sp_id,
                "name": p_info.get("fullName", "Unknown"),
            }
        else:
            result[side] = {"id": None, "name": "Unknown"}

    return result


# ─────────────────────────────────────────────
# Lineup Data
# ─────────────────────────────────────────────

def fetch_lineup_stats(game_pk: int, season: int) -> dict:
    """
    Get batting stats for each team's actual lineup in a game.
    Returns aggregated lineup quality metrics per side.
    """
    ck = f"lineup_{game_pk}"
    data = _mlb(f"game/{game_pk}/boxscore", cache_key=ck, cache_ttl_mins=360)

    result = {}
    for side in ["home", "away"]:
        team = data.get("teams", {}).get(side, {})
        batters = team.get("batters", [])[:9]  # starting 9
        players = team.get("players", {})

        avgs, obps, slugs = [], [], []
        for bid in batters:
            key = f"ID{bid}"
            p = players.get(key, {})
            s = p.get("seasonStats", {}).get("batting", {})
            try:
                avgs.append(float(s.get("avg", ".250").replace(".", "0.", 1)
                                  if str(s.get("avg","")).startswith(".") else s.get("avg", 0.250)))
            except Exception:
                avgs.append(0.250)
            try:
                obps.append(float(str(s.get("obp", "0.320")).lstrip(".")))
            except Exception:
                obps.append(0.320)
            try:
                slugs.append(float(str(s.get("slg", "0.400")).lstrip(".")))
            except Exception:
                slugs.append(0.400)

        result[side] = {
            "lineup_avg":  np.mean(avgs) if avgs else 0.250,
            "lineup_obp":  np.mean(obps) if obps else 0.320,
            "lineup_slg":  np.mean(slugs) if slugs else 0.400,
            "lineup_ops":  (np.mean(obps) + np.mean(slugs)) if obps else 0.720,
            "lineup_depth": len(batters),
        }

    return result


# ─────────────────────────────────────────────
# Upcoming Games (Today's Slate)
# ─────────────────────────────────────────────

def fetch_today_schedule() -> pd.DataFrame:
    """Fetch today's MLB schedule with probable pitchers."""
    today = date.today().strftime("%Y-%m-%d")
    data = _mlb("schedule", {
        "sportId": 1,
        "date": today,
        "gameType": "R",
        "hydrate": "probablePitcher,linescore,team",
    }, cache_key=f"today_{today}", cache_ttl_mins=30)

    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                continue

            home = g["teams"]["home"]
            away = g["teams"]["away"]

            def sp(team_key):
                pp = g["teams"][team_key].get("probablePitcher", {})
                return {"id": pp.get("id"), "name": pp.get("fullName", "TBD")}

            rows.append({
                "game_pk":      g["gamePk"],
                "game_date":    g["gameDate"][:10],
                "status":       status,
                "home_team":    home["team"]["name"],
                "away_team":    away["team"]["name"],
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_sp_id":   sp("home")["id"],
                "home_sp_name": sp("home")["name"],
                "away_sp_id":   sp("away")["id"],
                "away_sp_name": sp("away")["name"],
            })

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} games today ({today})")
    return df


# ─────────────────────────────────────────────
# Live Odds via The Odds API
# ─────────────────────────────────────────────

def fetch_mlb_odds() -> pd.DataFrame:
    """
    Fetch current MLB moneyline odds from The Odds API.
    Matches games by home/away team name.
    """
    if not ODDS_KEY:
        print("  No ODDS_API_KEY — skipping live odds. Add to .env.")
        return pd.DataFrame()

    try:
        params = {
            "apiKey":     ODDS_KEY,
            "regions":    "us",
            "markets":    "h2h",
            "oddsFormat": "american",
        }
        resp = requests.get(f"{ODDS_API}/sports/baseball_mlb/odds",
                            params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  Odds API → {resp.status_code} | quota remaining: {remaining}")
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()

    rows = []
    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        home_odds_list, away_odds_list = [], []

        for book in g.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    if outcome["name"] == home:
                        home_odds_list.append(outcome["price"])
                    elif outcome["name"] == away:
                        away_odds_list.append(outcome["price"])

        if not home_odds_list or not away_odds_list:
            continue

        # Best available line (highest payout = most favourable to bettor)
        best_home = max(home_odds_list, key=lambda x: x if x > 0 else 10000 / abs(x))
        best_away = max(away_odds_list, key=lambda x: x if x > 0 else 10000 / abs(x))
        # Consensus (median) line
        med_home = float(np.median(home_odds_list))
        med_away = float(np.median(away_odds_list))

        rows.append({
            "home_team":        home,
            "away_team":        away,
            "home_odds_best":   best_home,
            "away_odds_best":   best_away,
            "home_odds_median": med_home,
            "away_odds_median": med_away,
            "n_books":          len(g.get("bookmakers", [])),
            "commence_time":    g.get("commence_time", ""),
        })

    df = pd.DataFrame(rows)
    print(f"  Fetched odds for {len(df)} MLB games")
    return df


# ─────────────────────────────────────────────
# Park Factors
# ─────────────────────────────────────────────

# Static 2023/24 park factors (runs, normalised: 100 = neutral)
# Source: Baseball Reference park factors
PARK_FACTORS = {
    "Colorado Rockies":       115,  # Coors Field — extreme hitter's park
    "Boston Red Sox":         108,
    "Cincinnati Reds":        107,
    "Houston Astros":         97,
    "Texas Rangers":          103,
    "New York Yankees":       104,
    "Chicago Cubs":           103,
    "Philadelphia Phillies":  103,
    "San Francisco Giants":   96,
    "Los Angeles Dodgers":    96,
    "Seattle Mariners":       94,
    "Oakland Athletics":      97,
    "San Diego Padres":       94,
    "Miami Marlins":          93,
    "Tampa Bay Rays":         97,
    "Minnesota Twins":        102,
    "Cleveland Guardians":    98,
    "Detroit Tigers":         99,
    "Chicago White Sox":      101,
    "Kansas City Royals":     101,
    "St. Louis Cardinals":    98,
    "Milwaukee Brewers":      97,
    "Pittsburgh Pirates":     99,
    "Atlanta Braves":         100,
    "Washington Nationals":   101,
    "New York Mets":          98,
    "Baltimore Orioles":      103,
    "Toronto Blue Jays":      103,
    "Los Angeles Angels":     99,
    "Arizona Diamondbacks":   101,
}

def get_park_factor(home_team: str) -> float:
    """Return park factor (100 = neutral, >100 = hitter friendly)."""
    return PARK_FACTORS.get(home_team, 100) / 100.0


# ─────────────────────────────────────────────
# Head-to-Head Records
# ─────────────────────────────────────────────

def compute_h2h(schedule_df: pd.DataFrame, window_games: int = 20) -> dict:
    """
    Build a lookup of recent head-to-head win rates.
    Returns dict: (home_team, away_team) -> home_win_rate
    """
    h2h = {}
    df = schedule_df.sort_values("game_date")

    matchups = df.groupby(["home_team", "away_team"])
    for (home, away), games in matchups:
        recent = games.tail(window_games)
        h2h[(home, away)] = recent["home_win"].mean()

    return h2h
