#!/usr/bin/env python3
"""
Debug script to inspect the actual odds being passed to evaluate_spread.
Run this in --mode live to see what's happening with the Dodgers game.
"""
import sys
sys.path.insert(0, '.')

from mlb_data import fetch_today_schedule, fetch_mlb_odds
import pandas as pd

print("Fetching today's schedule and odds...")
today_df = fetch_today_schedule()
odds_df = fetch_mlb_odds()

print(f"\nFound {len(today_df)} games on schedule")
print(f"Found {len(odds_df)} games with odds\n")

# Look for Dodgers game
dodgers_games = odds_df[
    (odds_df['home_team'].str.contains('Dodgers', case=False, na=False)) |
    (odds_df['away_team'].str.contains('Dodgers', case=False, na=False))
]

if dodgers_games.empty:
    print("No Dodgers game found in odds data")
else:
    print("=" * 80)
    print("DODGERS GAME ODDS")
    print("=" * 80)
    for _, game in dodgers_games.iterrows():
        print(f"\nMatchup: {game['away_team']} @ {game['home_team']}")
        print(f"\nMoneyline:")
        print(f"  Home: {game['home_odds']:+.0f}")
        print(f"  Away: {game['away_odds']:+.0f}")
        print(f"\nRun Line:")
        print(f"  Home {game.get('rl_home_line', 'N/A')}: {game.get('rl_home_odds', 'N/A')}")
        print(f"  Away {game.get('rl_away_line', 'N/A')}: {game.get('rl_away_odds', 'N/A')}")
        print(f"\nBooks: {game.get('n_books', 'N/A')}")

print("\n" + "=" * 80)
print("ALL GAMES WITH BOTH ML AND RL ODDS")
print("=" * 80)

complete = odds_df[odds_df['rl_home_odds'].notna()].copy()
print(f"\n{len(complete)} games have complete odds\n")

for _, game in complete.iterrows():
    # Calculate what the model would show
    # We need to reverse-engineer what model prob would give the displayed edge
    print(f"{game['away_team']} @ {game['home_team']}")
    print(f"  ML: {game['home_odds']:+4.0f} / {game['away_odds']:+4.0f}")
    print(f"  RL: {game['rl_home_line']:+4.1f} {game['rl_home_odds']:+4.0f} / {game['rl_away_line']:+4.1f} {game['rl_away_odds']:+4.0f}")
    print()
