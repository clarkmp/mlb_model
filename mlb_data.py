"""
mlb_data.py
-----------
All data fetching for the MLB model.

Sources:
  - MLB Stats API (statsapi.mlb.com) — free, no key needed
    * Schedule / results, team stats, pitcher stats, lineups
  - The Odds API — live odds (requires ODDS_API_KEY in .env)
    * Bookmaker filter: FLIFF only (bookmaker key: "fliff")
    * Markets fetched: h2h (moneyline), spreads (run line)
    * Note: F5 odds are not a separate Odds API market — F5 bets use
    *       the moneyline adjusted by SP ERA differential
    * Player props: batter_home_runs
"""

import os, json, time, requests
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

FINAL_STATES = {"Final", "Game Over", "Completed Early", "F", "FT", "FR"}



# ─────────────────────────────────────────────
# Core HTTP helper
# ─────────────────────────────────────────────

def _get(url, params=None, cache_key="", cache_ttl_mins=60):
    if cache_key:
        cf = CACHE_DIR / f"{cache_key}.json"
        if cf.exists() and (time.time() - cf.stat().st_mtime) / 60 < cache_ttl_mins:
            with open(cf) as f:
                return json.load(f)
    resp = requests.get(url, params=params or {}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if cache_key:
        with open(CACHE_DIR / f"{cache_key}.json", "w") as f:
            json.dump(data, f)
    return data

def _mlb(endpoint, params=None, cache_key="", cache_ttl_mins=360):
    return _get(f"{MLB_API}/{endpoint}", params, cache_key, cache_ttl_mins)

def _odds_api(endpoint, params, cache_key="", cache_ttl_mins=30):
    """Odds API call — always injects the API key."""
    if not ODDS_KEY:
        raise ValueError("ODDS_API_KEY not set in .env")
    params = {**params, "apiKey": ODDS_KEY}
    data = _get(f"{ODDS_API}/{endpoint}", params, cache_key, cache_ttl_mins)
    return data


# ─────────────────────────────────────────────
# Schedule & Results
# ─────────────────────────────────────────────

def fetch_schedule(start_date, end_date, debug=False):
    ck   = f"schedule_{start_date}_{end_date}"
    data = _mlb("schedule", {
        "sportId": 1, "startDate": start_date, "endDate": end_date,
        "gameType": "R", "hydrate": "linescore,decisions,probablePitcher",
    }, cache_key=ck, cache_ttl_mins=720)

    dates_list   = data.get("dates", [])
    all_games_raw = sum(len(d.get("games", [])) for d in dates_list)

    if debug or all_games_raw == 0:
        print(f"    API: {len(dates_list)} date(s), {all_games_raw} game(s) "
              f"for {start_date}→{end_date}")

    rows = []
    for day in dates_list:
        for g in day.get("games", []):
            s        = g.get("status", {})
            abstract = s.get("abstractGameState", "")
            detailed = s.get("detailedState", "")
            coded    = s.get("codedGameState", "")
            is_final = (
                abstract in FINAL_STATES or detailed in FINAL_STATES
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

            home_score = (home.get("score")
                          or ls.get("teams", {}).get("home", {}).get("runs", 0) or 0)
            away_score = (away.get("score")
                          or ls.get("teams", {}).get("away", {}).get("runs", 0) or 0)
            home_win   = home.get("isWinner")
            if home_win is None:
                home_win = (home_score > away_score) if home_score != away_score else None
            if home_win is None:
                continue

            def sp(side):
                return (g["teams"][side].get("probablePitcher", {})
                        .get("fullName", "TBD"))

            # F5 score (first 5 innings) from linescore if available
            innings = ls.get("innings", [])
            home_f5 = sum(inn.get("home", {}).get("runs", 0)
                          for inn in innings[:5] if isinstance(inn.get("home"), dict))
            away_f5 = sum(inn.get("away", {}).get("runs", 0)
                          for inn in innings[:5] if isinstance(inn.get("away"), dict))

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
                "home_f5":      home_f5,
                "away_f5":      away_f5,
                "home_f5_win":  int(home_f5 > away_f5) if home_f5 + away_f5 > 0 else None,
                "innings":      ls.get("currentInning", 9),
                "home_sp":      sp("home"),
                "away_sp":      sp("away"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  {start_date} → {end_date} : "
          f"{len(rows)} completed / {all_games_raw} total")
    return df


def fetch_season_schedule(seasons=None, chunk_days=30):
    if seasons is None:
        seasons = [2022, 2023, 2024]
    all_dfs = []
    for season in seasons:
        print(f"\n  ── Season {season} ──")
        cursor      = date(season, 4, 1)
        season_end  = date(season, 10, 1)
        season_rows = 0
        while cursor < season_end:
            chunk_end = min(cursor + timedelta(days=chunk_days), season_end)
            chunk_df  = fetch_schedule(cursor.strftime("%Y-%m-%d"),
                                       chunk_end.strftime("%Y-%m-%d"))
            if not chunk_df.empty:
                all_dfs.append(chunk_df)
                season_rows += len(chunk_df)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.2)
        print(f"  Season {season} total: {season_rows} games")

    if not all_dfs:
        print("\n  WARNING: 0 games fetched. Run debug_api.py to diagnose.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("game_pk")
    print(f"\n  Grand total: {len(combined)} games across {len(seasons)} season(s)")
    return combined


# ─────────────────────────────────────────────
# Team season stats
# ─────────────────────────────────────────────

def fetch_team_season_stats(season):
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

    bat = _mlb("teams/stats", {"stats": "season", "group": "hitting",
               "season": season, "sportId": 1},
               cache_key=f"team_bat_{season}", cache_ttl_mins=1440)
    pit = _mlb("teams/stats", {"stats": "season", "group": "pitching",
               "season": season, "sportId": 1},
               cache_key=f"team_pit_{season}", cache_ttl_mins=1440)

    bat_rows = parse(bat, "bat")
    pit_rows = parse(pit, "pit")
    merged   = [{**bat_rows.get(tid, {}), **pit_rows.get(tid, {}),
                 "season": season}
                for tid in set(bat_rows) | set(pit_rows)]
    df = pd.DataFrame(merged)
    print(f"  Team stats: {len(df)} teams for {season}")
    return df


# ─────────────────────────────────────────────
# Pitcher stats
# ─────────────────────────────────────────────

def fetch_pitcher_stats(pitcher_id, season):
    try:
        data   = _mlb(f"people/{pitcher_id}/stats",
                      {"stats": "season", "group": "pitching", "season": season},
                      cache_key=f"pitcher_{pitcher_id}_{season}",
                      cache_ttl_mins=1440)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        s = splits[0].get("stat", {})
        def _f(key, default):
            v = str(s.get(key, default))
            v = v.lstrip(".")
            try:
                return float(f"0.{v}") if len(v) <= 3 and "." not in v else float(v)
            except Exception:
                return default
        return {
            "era":     _f("era",   4.50),
            "whip":    _f("whip",  1.30),
            "k_per_9": _f("strikeoutsPer9Inn", 8.0),
            "bb_per_9":_f("walksPer9Inn", 3.5),
            "fip":     _f("fieldingIndependent", 4.50),
            "innings": _f("inningsPitched", 0),
            "hr_per_9":_f("homeRunsPer9", 1.2),
            "opp_avg": _f("avg", 0.250),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
# Lineup stats
# ─────────────────────────────────────────────

def fetch_lineup_stats(game_pk, season):
    LEAGUE_AVG = {"lineup_avg": 0.250, "lineup_obp": 0.320,
                  "lineup_slg": 0.400, "lineup_ops": 0.720, "lineup_depth": 9}
    try:
        data = _mlb(f"game/{game_pk}/boxscore",
                    cache_key=f"lineup_{game_pk}", cache_ttl_mins=360)
    except Exception:
        return {"home": dict(LEAGUE_AVG), "away": dict(LEAGUE_AVG)}

    result = {}
    for side in ["home", "away"]:
        team    = data.get("teams", {}).get(side, {})
        batters = team.get("batters", [])[:9]
        players = team.get("players", {})
        avgs, obps, slugs = [], [], []
        for bid in batters:
            p = players.get(f"ID{bid}", {})
            s = p.get("seasonStats", {}).get("batting", {})
            def _s(field, default):
                raw = str(s.get(field, default)).lstrip(".")
                try:
                    return float(f"0.{raw}") if len(raw) <= 3 and "." not in raw else float(raw)
                except Exception:
                    return default
            avgs.append(_s("avg", 0.250))
            obps.append(_s("obp", 0.320))
            slugs.append(_s("slg", 0.400))
        if not avgs:
            result[side] = dict(LEAGUE_AVG)
        else:
            obp = float(np.mean(obps))
            slg = float(np.mean(slugs))
            result[side] = {
                "lineup_avg":   round(float(np.mean(avgs)), 4),
                "lineup_obp":   round(obp, 4),
                "lineup_slg":   round(slg, 4),
                "lineup_ops":   round(obp + slg, 4),
                "lineup_depth": len(batters),
            }
    return result


# ─────────────────────────────────────────────
# Today's schedule
# ─────────────────────────────────────────────

def fetch_today_schedule():
    today = date.today().strftime("%Y-%m-%d")
    data  = _mlb("schedule", {
        "sportId": 1, "date": today, "gameType": "R",
        "hydrate": "probablePitcher,linescore,team",
    }, cache_key=f"today_{today}", cache_ttl_mins=30)

    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            abstract = g.get("status", {}).get("abstractGameState", "")
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
# Odds API — all markets from FLIFF only
# ─────────────────────────────────────────────

def _parse_h2h(bookmakers, home, away):
    """Extract best moneyline from a list of bookmaker entries."""
    h_list, a_list = [], []
    for book in bookmakers:
        for mkt in book.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            for o in mkt["outcomes"]:
                if o["name"] == home:
                    h_list.append(o["price"])
                elif o["name"] == away:
                    a_list.append(o["price"])
    if not h_list or not a_list:
        return None, None
    best = lambda lst: max(lst, key=lambda x: x if x > 0 else 10000/abs(x))
    return best(h_list), best(a_list)


def _parse_spread(bookmakers, home, away):
    """Extract run-line (spread) odds. Standard MLB run line is -1.5/+1.5."""
    for book in bookmakers:
        for mkt in book.get("markets", []):
            if mkt["key"] != "spreads":
                continue
            home_line = away_line = None
            home_odds_rl = away_odds_rl = None
            for o in mkt["outcomes"]:
                if o["name"] == home:
                    home_line    = o.get("point", -1.5)
                    home_odds_rl = o["price"]
                elif o["name"] == away:
                    away_line    = o.get("point", 1.5)
                    away_odds_rl = o["price"]
            if home_odds_rl and away_odds_rl:
                return home_line, home_odds_rl, away_line, away_odds_rl
    return None, None, None, None


def _parse_f5(bookmakers, home, away):
    """
    F5 innings is not a standard Odds API market for MLB.
    Returns None, None — F5 recommendations are derived in mlb_betting.py
    using the full-game moneyline odds adjusted by SP ERA differential.
    """
    return None, None


def fetch_mlb_odds():
    """
    Fetch moneyline + spread + F5 odds from The Odds API (all US bookmakers).
    Returns a DataFrame with one row per game containing all markets.
    """
    if not ODDS_KEY:
        print("  No ODDS_API_KEY in .env — skipping live odds.")
        return pd.DataFrame()

    all_markets = "h2h,spreads"
    try:
        resp = requests.get(
            f"{ODDS_API}/sports/baseball_mlb/odds",
            params={
                "apiKey":     ODDS_KEY,
                "regions":    "us",
                "markets":    all_markets,
                "oddsFormat": "american",
            },
            timeout=20,
        )
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  Odds API → {resp.status_code} | quota remaining: {remaining}")
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()

    if not games:
        print("  No odds returned from Odds API.")
        return pd.DataFrame()

    rows = []
    for g in games:
        home  = g["home_team"]
        away  = g["away_team"]
        books = g.get("bookmakers", [])

        h2h_home, h2h_away = _parse_h2h(books, home, away)
        rl_home_pts, rl_home_odds, rl_away_pts, rl_away_odds = _parse_spread(books, home, away)
        f5_home, f5_away = _parse_f5(books, home, away)

        if h2h_home is None:
            continue

        rows.append({
            "home_team":        home,
            "away_team":        away,
            "commence_time":    g.get("commence_time", ""),
            "n_books":          len(books),
            # Moneyline
            "home_odds":        h2h_home,
            "away_odds":        h2h_away,
            # Run line (spread)
            "rl_home_line":     rl_home_pts,
            "rl_home_odds":     rl_home_odds,
            "rl_away_line":     rl_away_pts,
            "rl_away_odds":     rl_away_odds,
            # First 5 innings — derived from ML odds + SP ERA, not a separate market
            "f5_home_odds":     None,
            "f5_away_odds":     None,
        })

    df = pd.DataFrame(rows)
    print(f"  Fetched odds for {len(df)} game(s) | "
          f"spreads: {df['rl_home_odds'].notna().sum()} | "
          f"F5: {df['f5_home_odds'].notna().sum()}")
    return df


# ─────────────────────────────────────────────
# Home run props (batter_home_runs market)
# ─────────────────────────────────────────────

def fetch_hr_props(game_id: str) -> pd.DataFrame:
    """
    Fetch batter home run props for a specific game from FLIFF.
    game_id is the Odds API event ID (from fetch_mlb_odds raw response).
    Returns DataFrame with player, line, over_odds, under_odds.
    """
    if not ODDS_KEY:
        return pd.DataFrame()
    try:
        resp = requests.get(
            f"{ODDS_API}/sports/baseball_mlb/events/{game_id}/odds",
            params={
                "apiKey":     ODDS_KEY,
                "regions":    "us",
                "markets":    "batter_home_runs",
                "oddsFormat": "american",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data  = resp.json()
        books = data.get("bookmakers", [])
    except Exception as e:
        print(f"  HR props error for {game_id}: {e}")
        return pd.DataFrame()

    rows = []
    for book in books:
        for mkt in book.get("markets", []):
            if mkt["key"] != "batter_home_runs":
                continue
            outcomes = mkt.get("outcomes", [])
            # Group by player name
            player_map: dict = {}
            for o in outcomes:
                player = o.get("description", o.get("name", "Unknown"))
                side   = o["name"].lower()   # "Over" or "Under"
                price  = o["price"]
                line   = o.get("point", 0.5)
                if player not in player_map:
                    player_map[player] = {"player": player, "line": line,
                                          "over_odds": None, "under_odds": None}
                if "over" in side:
                    player_map[player]["over_odds"] = price
                else:
                    player_map[player]["under_odds"] = price
            rows.extend(player_map.values())

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["over_odds"])
    return df


# ─────────────────────────────────────────────
# Event ID lookup (needed for props)
# ─────────────────────────────────────────────

def fetch_event_ids() -> dict:
    """
    Returns {(home_team, away_team): event_id} for today's games.
    Used to fetch props for a specific game.
    """
    if not ODDS_KEY:
        return {}
    try:
        resp = requests.get(
            f"{ODDS_API}/sports/baseball_mlb/events",
            params={"apiKey": ODDS_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception:
        return {}
    return {(e["home_team"], e["away_team"]): e["id"] for e in events}


# ─────────────────────────────────────────────
# Park factors & H2H
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

def get_park_factor(home_team):
    return PARK_FACTORS.get(home_team, 1.00)

def compute_h2h(schedule_df, window_games=20):
    h2h = {}
    df  = schedule_df.sort_values("game_date")
    for (home, away), games in df.groupby(["home_team", "away_team"]):
        h2h[(home, away)] = games.tail(window_games)["home_win"].mean()
    return h2h
