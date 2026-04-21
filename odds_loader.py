"""
Historical MLB odds loader and scraper integration.

Downloads the 76MB bulk dataset from GitHub (2021-04-01 to 2025-08-16).
Optionally scrapes new games using the included scraper for 2026 season.
Merges everything into a DataFrame keyed by (game_date, home_team, away_team).

Usage in main.py:
    from odds_loader import load_historical_odds
    odds_df = load_historical_odds([2023, 2024, 2025, 2026])
"""

import json
import os
import sys
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import shutil

# Paths
CACHE_DIR = Path("cache")
BULK_DATASET_URL = "https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/download/dataset/mlb_odds.json"
BULK_DATASET_CACHE = CACHE_DIR / "bulk_odds_2021_2025.json"
SCRAPED_ODDS_CACHE = CACHE_DIR / "scraped_odds_2026.json"
MERGED_ODDS_CACHE = CACHE_DIR / "merged_historical_odds.pkl"

# Scraper files (copied from upload)
SCRAPER_DIR = Path(__file__).parent / "scraper"
SCRAPER_SCRIPT = SCRAPER_DIR / "scraper.py"
POOLS_SCRIPT = SCRAPER_DIR / "pools.py"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)


def _download_bulk_dataset():
    """
    Use the uploaded bulk dataset from /mnt/user-data/uploads or cache.
    Does NOT attempt GitHub download (URL returns 404).
    """
    _ensure_cache_dir()
    
    # Check cache first
    if BULK_DATASET_CACHE.exists():
        print(f"  Bulk dataset already cached at {BULK_DATASET_CACHE}")
        return BULK_DATASET_CACHE
    
    # Use uploaded file
    uploaded_path = Path("/mnt/user-data/uploads/mlb_odds_dataset.json")
    if uploaded_path.exists():
        print(f"  Using uploaded bulk dataset ({uploaded_path.stat().st_size // (1024*1024)} MB)...")
        print(f"  Copying to cache...")
        import shutil
        shutil.copy(uploaded_path, BULK_DATASET_CACHE)
        print(f"  ✓ Cached to {BULK_DATASET_CACHE}")
        return BULK_DATASET_CACHE
    
    # No uploaded file found
    print(f"  ERROR: Bulk dataset not found at {uploaded_path}")
    print(f"  Please upload mlb_odds_dataset.json to /mnt/user-data/uploads/")
    return None


def _parse_bulk_dataset(json_path):
    """
    Parse the bulk dataset JSON into a DataFrame.
    
    Returns DataFrame with columns:
        game_date, home_team, away_team, home_score, away_score,
        home_ml_open, away_ml_open, home_ml_close, away_ml_close,
        home_rl_open, away_rl_open, home_rl_close, away_rl_close,
        rl_line (always ±1.5 for MLB)
    
    Aggregates across sportsbooks using median.
    """
    print(f"  Parsing bulk dataset from {json_path}...")
    
    with open(json_path) as f:
        data = json.load(f)
    
    rows = []
    
    for game_date, games in data.items():
        for game in games:
            game_view = game.get("gameView", {})
            odds = game.get("odds", {})
            
            home_team = game_view.get("homeTeam", {}).get("fullName")
            away_team = game_view.get("awayTeam", {}).get("fullName")
            home_score = game_view.get("homeTeamScore")
            away_score = game_view.get("awayTeamScore")
            
            if not home_team or not away_team:
                continue
            
            # ── Moneyline odds ────────────────────────────────────────────
            ml_odds = odds.get("moneyline", [])
            home_ml_open_list, away_ml_open_list = [], []
            home_ml_close_list, away_ml_close_list = [], []
            
            for book in ml_odds:
                ol = book.get("openingLine", {})
                cl = book.get("currentLine", {})
                
                if ol.get("homeOdds") is not None:
                    home_ml_open_list.append(ol["homeOdds"])
                if ol.get("awayOdds") is not None:
                    away_ml_open_list.append(ol["awayOdds"])
                if cl.get("homeOdds") is not None:
                    home_ml_close_list.append(cl["homeOdds"])
                if cl.get("awayOdds") is not None:
                    away_ml_close_list.append(cl["awayOdds"])
            
            # Use median across books (most robust to outliers)
            home_ml_open = pd.Series(home_ml_open_list).median() if home_ml_open_list else None
            away_ml_open = pd.Series(away_ml_open_list).median() if away_ml_open_list else None
            home_ml_close = pd.Series(home_ml_close_list).median() if home_ml_close_list else None
            away_ml_close = pd.Series(away_ml_close_list).median() if away_ml_close_list else None
            
            # ── Run line (point spread) odds ──────────────────────────────
            ps_odds = odds.get("pointspread", [])
            home_rl_open_list, away_rl_open_list = [], []
            home_rl_close_list, away_rl_close_list = [], []
            rl_line = None
            
            for book in ps_odds:
                ol = book.get("openingLine", {})
                cl = book.get("currentLine", {})
                
                # MLB run line is always ±1.5
                if ol.get("homeSpread") is not None:
                    rl_line = abs(ol["homeSpread"])  # Should be 1.5
                
                if ol.get("homeOdds") is not None:
                    home_rl_open_list.append(ol["homeOdds"])
                if ol.get("awayOdds") is not None:
                    away_rl_open_list.append(ol["awayOdds"])
                if cl.get("homeOdds") is not None:
                    home_rl_close_list.append(cl["homeOdds"])
                if cl.get("awayOdds") is not None:
                    away_rl_close_list.append(cl["awayOdds"])
            
            home_rl_open = pd.Series(home_rl_open_list).median() if home_rl_open_list else None
            away_rl_open = pd.Series(away_rl_open_list).median() if away_rl_open_list else None
            home_rl_close = pd.Series(home_rl_close_list).median() if home_rl_close_list else None
            away_rl_close = pd.Series(away_rl_close_list).median() if away_rl_close_list else None
            
            rows.append({
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_ml_open": home_ml_open,
                "away_ml_open": away_ml_open,
                "home_ml_close": home_ml_close,
                "away_ml_close": away_ml_close,
                "home_rl_open": home_rl_open,
                "away_rl_open": away_rl_open,
                "home_rl_close": home_rl_close,
                "away_rl_close": away_rl_close,
                "rl_line": rl_line if rl_line else 1.5,
            })
    
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    
    print(f"  Parsed {len(df)} games from bulk dataset")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    
    return df


def _setup_scraper():
    """Copy scraper files to a local directory so they can be imported/run."""
    SCRAPER_DIR.mkdir(exist_ok=True)
    
    # Copy scraper.py and pools.py from uploads to scraper/
    upload_dir = Path("/mnt/user-data/uploads")
    
    if not (upload_dir / "scraper.py").exists():
        print("  WARNING: scraper.py not found in uploads — scraping disabled")
        return False
    
    shutil.copy(upload_dir / "scraper.py", SCRAPER_SCRIPT)
    shutil.copy(upload_dir / "pools.py", POOLS_SCRIPT)
    
    print(f"  Scraper files copied to {SCRAPER_DIR}")
    return True


def _scrape_new_games(start_date, end_date):
    """
    Run the scraper to fetch games between start_date and end_date.
    
    Returns path to scraped JSON or None if scraping fails.
    """
    if not SCRAPER_SCRIPT.exists():
        if not _setup_scraper():
            return None
    
    output_file = SCRAPED_ODDS_CACHE
    
    print(f"  Scraping new games from {start_date} to {end_date}...")
    print(f"  (This may take 1-2 minutes per week of games)")
    
    try:
        # Check if aiohttp is installed
        try:
            import aiohttp
        except ImportError:
            print("  ERROR: aiohttp not installed. Run: pip install aiohttp tqdm")
            return None
        
        # Run scraper as subprocess
        cmd = [
            sys.executable,
            str(SCRAPER_SCRIPT),
            start_date,
            end_date,
            "-t", "moneyline", "pointspread",
            "-c", "5",  # 5 concurrent — safe default
            "-o", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"  Scraper failed: {result.stderr}")
            return None
        
        if not output_file.exists():
            print("  Scraper completed but no output file created")
            return None
        
        print(f"  Scraping complete → {output_file}")
        return output_file
    
    except subprocess.TimeoutExpired:
        print("  Scraper timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"  Scraping error: {e}")
        return None


def load_historical_odds(seasons, enable_scraping=False):
    """
    Load historical odds for the given seasons.
    
    Args:
        seasons: list of years (e.g. [2023, 2024, 2025, 2026])
        enable_scraping: if True, scrape new 2026 games beyond bulk dataset
    
    Returns:
        DataFrame with columns matching _parse_bulk_dataset output.
        Returns empty DataFrame if loading fails.
    """
    _ensure_cache_dir()
    
    # ── Check if merged cache exists and is recent ───────────────────────
    if MERGED_ODDS_CACHE.exists():
        cache_age = datetime.now().timestamp() - MERGED_ODDS_CACHE.stat().st_mtime
        if cache_age < 86400:  # 24 hours
            print(f"  Using cached historical odds (age: {cache_age/3600:.1f}h)")
            return pd.read_pickle(MERGED_ODDS_CACHE)
    
    # ── Load bulk dataset ─────────────────────────────────────────────────
    bulk_path = _download_bulk_dataset()
    if not bulk_path:
        print("  Failed to load bulk dataset — returning empty DataFrame")
        return pd.DataFrame()
    
    bulk_df = _parse_bulk_dataset(bulk_path)
    
    # ── Optionally scrape new games ───────────────────────────────────────
    if enable_scraping:
        # Determine what needs scraping
        bulk_end = bulk_df["game_date"].max().date()
        today = datetime.now().date()
        
        if today > bulk_end:
            scrape_start = (bulk_end + timedelta(days=1)).strftime("%Y-%m-%d")
            scrape_end = today.strftime("%Y-%m-%d")
            
            scraped_path = _scrape_new_games(scrape_start, scrape_end)
            
            if scraped_path:
                scraped_df = _parse_bulk_dataset(scraped_path)
                print(f"  Scraped {len(scraped_df)} new games")
                bulk_df = pd.concat([bulk_df, scraped_df], ignore_index=True)
    
    # ── Filter to requested seasons ───────────────────────────────────────
    bulk_df["year"] = bulk_df["game_date"].dt.year
    filtered = bulk_df[bulk_df["year"].isin(seasons)].copy()
    filtered = filtered.drop(columns=["year"])
    
    print(f"  Filtered to {len(filtered)} games across seasons {seasons}")
    
    # ── Cache the merged result ───────────────────────────────────────────
    filtered.to_pickle(MERGED_ODDS_CACHE)
    print(f"  Cached to {MERGED_ODDS_CACHE}")
    
    return filtered


def clear_odds_cache():
    """Delete all cached odds files to force fresh download."""
    for cache_file in [BULK_DATASET_CACHE, SCRAPED_ODDS_CACHE, MERGED_ODDS_CACHE]:
        if cache_file.exists():
            cache_file.unlink()
            print(f"  Deleted {cache_file}")
    print("  Odds cache cleared")


if __name__ == "__main__":
    # Test the loader
    print("Testing odds loader...")
    df = load_historical_odds([2023, 2024, 2025], enable_scraping=False)
    print(f"\nLoaded {len(df)} games")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:")
    print(df.head(3).to_string())
    
    # Show coverage stats
    print(f"\nCoverage:")
    print(f"  Moneyline close: {df['home_ml_close'].notna().sum()} / {len(df)} games")
    print(f"  Run line close:  {df['home_rl_close'].notna().sum()} / {len(df)} games")
