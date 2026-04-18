"""
mlb_data.py
-----------
All data fetching for the MLB model.

Sources:
  - MLB Stats API (statsapi.mlb.com) — free, no key needed
  - The Odds API — live moneyline odds (requires ODDS_API_KEY in .env)

Fixes applied vs v1:
  - Removed the `fields` filter param that was stripping status/score fields
  - Robust status check: accepts 'Final', 'Game Over', 'Completed Early'
  - Chunked fetching (30-day windows) to avoid API response size limits
  - Better debug output so you can see exactly what the API returns
  - score fallback: reads from linescore if top-level score missing
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

MLB_API  = "https://statsapi.mlb.com/api/v1"
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_KEY = os.getenv("ODDS_API_KEY", "")

# Status strings the MLB API uses for completed games
FINAL_STATES = {"Final", "Game Over", "Completed Early", "F", "FT", "FR"}


# ─────────────────────────────────────────────
# Core HTTP helper
# ─────────────────────────────────────────────

def _get(url: str, params: dict = None, cache_key: str = "",
         cache_ttl_mins: int = 60) -> dict | list:
    """GET with optional file cache."""
    if cache_key:
        cf = CACHE_DIR / f"{cache_key}.json"
        if cf.exists():
            age = (time.time() - cf.stat().st_mtime) / 60
            if age < cache_ttl_mins:
                with open(cf) as f:
                    return json.load(f)

    resp = requests.get(url, params=params or {}, timeout=30)
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

def fetch_schedule(start_date: str, end_date: str,
                   debug: bool = False) -> pd.DataFrame:
    """
    Fetch completed MLB games between two dates.

    Key fixes vs old version:
      - NO `fields` param — it was silently stripping status/score data
      - Accepts all final-state strings, not just literal "Final"
      - Falls back to linescore for scores if top-level score is 0/missing
      - Prints a sample of what came back so you can diagnose issues
    """
    ck = f"schedule_{start_date}_{end_date}"

    # NOTE: Do NOT add a `fields` param here — it filters out status/score data
    data = _mlb("schedule", {
        "sportId":   1,
        "startDate": start_date,
        "endDate":   end_date,
        "gameType":  "R",
        "hydrate":   "linescore,decisions,probablePitcher",
    }, cache_key=ck, cache_ttl_mins=720)

    dates_list = data.get("dates", [])
    all_games_raw = sum(len(d.get("games", [])) for d in dates_list)

    if debug or all_games_raw == 0:
        print(f"    API returned {len(dates_list)} date(s), "
              f"{all_games_raw} raw game(s) for {start_date}→{end_date}")
        if dates_list:
            sample_g = dates_list[0]["games"][0] if dates_list[0].get("games") else {}
            print(f"    Sample game status: {sample_g.get('status', 'NO STATUS KEY')}")
            print(f"    Sample abstractGameState: "
                  f"{sample_g.get('status',{}).get('abstractGameState','MISSING')}")
            print(f"    Sample detailedState: "
                  f"{sample_g.get('status',{}).get('detailedState','MISSING')}")

    rows = []
    for day in dates_list:
        for g in day.get("games", []):
            status = g.get("status", {})
            abstract = status.get("abstractGameState", "")
            detailed = status.get("detailedState", "")
            coded    = status.get("codedGameState", "")

            # Accept any indicator of a finished game
            is_final = (
                abstract in FINAL_STATES
                or detailed in FINAL_STATES
                or coded in ("F", "FR", "FT")
                or "final" in abstract.lower()
                or "final" in detailed.lower()
                or "over"  in detailed.lower()
            )
            if not is_final:
                continue

            home = g["teams"]["home"]
            away = g["teams"]["away"]
            ls   = g.get("linescore", {})

            # Score: try top-level first, fall back to linescore
            home_score = (
                home.get("score")
                or ls.get("teams", {}).get("home", {}).get("runs", 0)
                or 0
            )
            away_score = (
                away.get("score")
                or ls.get("teams", {}).get("away", {}).get("runs", 0)
                or 0
            )

            # isWinner: try flag, fall back to score comparison
            home_win = home.get("isWinner")
            if home_win is None:
                home_win = (home_score > away_score) if home_score != away_score else None
            if home_win is None:
                continue  # Skip tie / incomplete score

            # Starting pitcher names
            def sp(side):
                return (
                    g["teams"][side]
                    .get("probablePitcher", {})
                    .get("fullName", "TBD")
                )

            rows.append({
                "game_pk":      g["gamePk"],
                "game_date":    g["gameDate"][:10],
                "home_team":    home["team"]["name"],
                "away_team":    away["team"]["name"],
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_score":   int(home_score),
                "away_score":   int(away_score),
                "home_win":     int(bool(home_win)),
                "innings":      ls.get("currentInning", 9),
                "home_sp":      sp("home"),
                "away_sp":      sp("away"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  {start_date} → {end_date} : "
          f"{len(rows)} completed / {all_games_raw} total games in API response")
    return df


def fetch_season_schedule(seasons: list[int] = None,
                          chunk_days: int = 30) -> pd.DataFrame:
    """
    Fetch full season(s) of completed games.
    Fetches in 30-day chunks to stay within API response limits.
    MLB regular season: roughly April 1 – October 1.
    """
    if seasons is None:
        seasons = [2023, 2024]

    all_dfs = []
    for season in seasons:
        print(f"\n  ── Season {season} ──")
        season_start = date(season, 4, 1)
        season_end   = date(season, 10, 1)
        cursor = season_start
        season_rows = 0

        while cursor < season_end:
            chunk_end = min(cursor + timedelta(days=chunk_days), season_end)
            chunk_df  = fetch_schedule(
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            if not chunk_df.empty:
                all_dfs.append(chunk_df)
                season_rows += len(chunk_df)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.2)   # be polite to the API

        print(f"  Season {season} total: {season_rows} games")

    if not all_dfs:
        print("\n  WARNING: 0 games fetched across all seasons.")
        print("  Try running debug_api.py on your VM to inspect the raw API response.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("game_pk")
    print(f"\n  Grand total: {len(combined)} games across {len(seasons)} season(s)")
    return combined


# ─────────────────────────────────────────────
# Team Season Stats
# ─────────────────────────────────────────────

def fetch_team_season_stats(season: int) -> pd.DataFrame:
    """Fetch team-level batting and pitching stats for a full season."""
    def parse(data, prefix):
        rows = {}
        for entry in data.get("stats", [{}])[0].get("splits", []):
            team = entry.get("team", {})
            tid  = team.get("id")
            if not tid:
                continue
            rows[tid] = {"team_id": tid, "team_name": team.get("name", "")}
            for k, v in entry.get("stat", {}).items():
                try:
                    rows[tid][f"{prefix}_{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
        return rows

    bat = _mlb("teams/stats", {
        "stats": "season", "group": "hitting",
        "season": season, "sportId": 1,
    }, cache_key=f"team_bat_{season}", cache_ttl_mins=1440)

    pit = _mlb("teams/stats", {
        "stats": "season", "group": "pitching",
        "season": season, "sportId": 1,
    }, cache_key=f"team_pit_{season}", cache_ttl_mins=1440)

    bat_rows = parse(bat, "bat")
    pit_rows = parse(pit, "pit")
    all_ids  = set(bat_rows) | set(pit_rows)
    merged   = [{**bat_rows.get(tid, {}), **pit_rows.get(tid, {}),
                 "season": season} for tid in all_ids]

    df = pd.DataFrame(merged)
    print(f"  Team stats: {len(df)} teams for {season}")
    return df


# ─────────────────────────────────────────────
# Pitcher stats
# ─────────────────────────────────────────────

def fetch_pitcher_stats(pitcher_id: int, season: int) -> dict:
    """Fetch season pitching stats for one pitcher."""
    try:
        data = _mlb(f"people/{pitcher_id}/stats", {
            "stats": "season", "group": "pitching", "season": season,
        }, cache_key=f"pitcher_{pitcher_id}_{season}", cache_ttl_mins=1440)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        s = splits[0].get("stat", {})

        def _f(key, default):
            v = s.get(key, default)
            try:
                # Handle ".250" style strings
                return float(str(v).lstrip(".").replace("..", "."))
            except Exception:
                return default

        return {
            "era":     _f("era", 4.50),
            "whip":    _f("whip", 1.30),
            "k_per_9": _f("strikeoutsPer9Inn", 8.0),
            "bb_per_9":_f("walksPer9Inn", 3.5),
            "fip":     _f("fieldingIndependent", 4.50),
            "innings": _f("inningsPitched", 0),
            "hr_per_9":_f("homeRunsPer9", 1.2),
            "opp_avg": _f("avg", 0.250),
        }
    except Exception as e:
        return {}


# ─────────────────────────────────────────────
# Today's schedule
# ─────────────────────────────────────────────

def fetch_today_schedule() -> pd.DataFrame:
    """Fetch today's MLB schedule with probable pitchers."""
    today = date.today().strftime("%Y-%m-%d")
    data  = _mlb("schedule", {
        "sportId":  1,
        "date":     today,
        "gameType": "R",
        "hydrate":  "probablePitcher,linescore,team",
    }, cache_key=f"today_{today}", cache_ttl_mins=30)

    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            abstract = g.get("status", {}).get("abstractGameState", "")
            # Skip already-finished games
            if abstract in FINAL_STATES or "final" in abstract.lower():
                continue

            home = g["teams"]["home"]
            away = g["teams"]["away"]

            def sp(side):
                pp = g["teams"][side].get("probablePitcher", {})
                return {"id": pp.get("id"), "name": pp.get("fullName", "TBD")}

            rows.append({
                "game_pk":      g["gamePk"],
                "game_date":    g["gameDate"][:10],
                "status":       abstract,
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
    print(f"  Found {len(df)} game(s) today ({today})")
    return df


# ─────────────────────────────────────────────
# Live odds via The Odds API
# ─────────────────────────────────────────────

def fetch_mlb_odds() -> pd.DataFrame:
    """Fetch current MLB moneyline odds. Requires ODDS_API_KEY in .env."""
    if not ODDS_KEY:
        print("  No ODDS_API_KEY in .env — skipping live odds.")
        return pd.DataFrame()

    try:
        resp = requests.get(
            f"{ODDS_API}/sports/baseball_mlb/odds",
            params={"apiKey": ODDS_KEY, "regions": "us",
                    "markets": "h2h", "oddsFormat": "american"},
            timeout=15,
        )
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  Odds API → {resp.status_code} | quota remaining: {remaining}")
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()

    rows = []
    for g in games:
        home, away = g["home_team"], g["away_team"]
        h_list, a_list = [], []
        for book in g.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                for o in mkt["outcomes"]:
                    if o["name"] == home:
                        h_list.append(o["price"])
                    elif o["name"] == away:
                        a_list.append(o["price"])
        if not h_list or not a_list:
            continue
        rows.append({
            "home_team":        home,
            "away_team":        away,
            "home_odds_best":   max(h_list, key=lambda x: x if x > 0 else 10000/abs(x)),
            "away_odds_best":   max(a_list, key=lambda x: x if x > 0 else 10000/abs(x)),
            "home_odds_median": float(np.median(h_list)),
            "away_odds_median": float(np.median(a_list)),
            "n_books":          len(g.get("bookmakers", [])),
            "commence_time":    g.get("commence_time", ""),
        })

    df = pd.DataFrame(rows)
    print(f"  Fetched odds for {len(df)} MLB game(s)")
    return df


# ─────────────────────────────────────────────
# Park factors (2023/24 Baseball Reference)
# ─────────────────────────────────────────────

PARK_FACTORS = {
    "Colorado Rockies":       1.15,
    "Boston Red Sox":         1.08,
    "Cincinnati Reds":        1.07,
    "Texas Rangers":          1.03,
    "Chicago Cubs":           1.03,
    "Philadelphia Phillies":  1.03,
    "New York Yankees":       1.04,
    "Baltimore Orioles":      1.03,
    "Toronto Blue Jays":      1.03,
    "Minnesota Twins":        1.02,
    "Chicago White Sox":      1.01,
    "Kansas City Royals":     1.01,
    "Washington Nationals":   1.01,
    "Arizona Diamondbacks":   1.01,
    "Los Angeles Angels":     0.99,
    "Detroit Tigers":         0.99,
    "Pittsburgh Pirates":     0.99,
    "Atlanta Braves":         1.00,
    "St. Louis Cardinals":    0.98,
    "Cleveland Guardians":    0.98,
    "New York Mets":          0.98,
    "Houston Astros":         0.97,
    "Oakland Athletics":      0.97,
    "Tampa Bay Rays":         0.97,
    "Milwaukee Brewers":      0.97,
    "San Francisco Giants":   0.96,
    "Los Angeles Dodgers":    0.96,
    "Seattle Mariners":       0.94,
    "San Diego Padres":       0.94,
    "Miami Marlins":          0.93,
}

def get_park_factor(home_team: str) -> float:
    return PARK_FACTORS.get(home_team, 1.00)


# ─────────────────────────────────────────────
# Head-to-head
# ─────────────────────────────────────────────

def compute_h2h(schedule_df: pd.DataFrame, window_games: int = 20) -> dict:
    """Build home-win-rate lookup for each (home, away) pairing."""
    h2h = {}
    df = schedule_df.sort_values("game_date")
    for (home, away), games in df.groupby(["home_team", "away_team"]):
        h2h[(home, away)] = games.tail(window_games)["home_win"].mean()
    return h2h


# ─────────────────────────────────────────────
# Lineup stats (batting quality for a specific game)
# ─────────────────────────────────────────────

def fetch_lineup_stats(game_pk: int, season: int) -> dict:
    """
    Get aggregated batting stats for each team's starting lineup in a game.
    Returns dict with keys 'home' and 'away', each containing:
        lineup_avg, lineup_obp, lineup_slg, lineup_ops, lineup_depth
    Falls back to league-average values if the boxscore isn't available yet.
    """
    LEAGUE_AVG = {
        "lineup_avg":   0.250,
        "lineup_obp":   0.320,
        "lineup_slg":   0.400,
        "lineup_ops":   0.720,
        "lineup_depth": 9,
    }

    try:
        data = _mlb(
            f"game/{game_pk}/boxscore",
            cache_key=f"lineup_{game_pk}",
            cache_ttl_mins=360,
        )
    except Exception:
        return {"home": dict(LEAGUE_AVG), "away": dict(LEAGUE_AVG)}

    result = {}
    for side in ["home", "away"]:
        team    = data.get("teams", {}).get(side, {})
        batters = team.get("batters", [])[:9]
        players = team.get("players", {})

        avgs, obps, slugs = [], [], []
        for bid in batters:
            key = f"ID{bid}"
            p   = players.get(key, {})
            s   = p.get("seasonStats", {}).get("batting", {})

            def _stat(field, default):
                raw = str(s.get(field, default))
                # MLB API returns batting avg as ".250" — strip leading dot
                raw = raw.lstrip(".")
                try:
                    return float(f"0.{raw}") if len(raw) <= 3 and "." not in raw else float(raw)
                except (ValueError, TypeError):
                    return default

            avgs.append(_stat("avg",  0.250))
            obps.append(_stat("obp",  0.320))
            slugs.append(_stat("slg", 0.400))

        if not avgs:
            result[side] = dict(LEAGUE_AVG)
        else:
            avg  = float(np.mean(avgs))
            obp  = float(np.mean(obps))
            slg  = float(np.mean(slugs))
            result[side] = {
                "lineup_avg":   round(avg, 4),
                "lineup_obp":   round(obp, 4),
                "lineup_slg":   round(slg, 4),
                "lineup_ops":   round(obp + slg, 4),
                "lineup_depth": len(batters),
            }

    return result
