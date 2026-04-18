"""
Run this on your VM to diagnose the 0 games issue.
It prints the raw API response for a small date window
so we can see exactly what fields are returned.

Usage: python debug_api.py
"""
import requests, json, sys

BASE = "https://statsapi.mlb.com/api/v1"

print("=" * 60)
print("TEST 1 — minimal params, NO fields filter")
print("=" * 60)
resp = requests.get(f"{BASE}/schedule", params={
    "sportId": 1,
    "startDate": "2024-04-01",
    "endDate":   "2024-04-03",
    "gameType":  "R",
}, timeout=15)
print("HTTP:", resp.status_code)
data = resp.json()
dates = data.get("dates", [])
print("Dates returned:", len(dates))
total_games = sum(len(d.get("games", [])) for d in dates)
print("Total games:", total_games)

if total_games > 0:
    # Print the first game's full structure
    g = dates[0]["games"][0]
    print("\nFirst game raw (trimmed):")
    # Show status structure
    print("  status:", json.dumps(g.get("status", {}), indent=4))
    print("  teams.home keys:", list(g.get("teams", {}).get("home", {}).keys()))
    print("  teams.away keys:", list(g.get("teams", {}).get("away", {}).keys()))
    print("  gameDate:", g.get("gameDate"))
    print("  gamePk:", g.get("gamePk"))
    home = g["teams"]["home"]
    print("  home.score:", home.get("score"))
    print("  home.isWinner:", home.get("isWinner"))

print()
print("=" * 60)
print("TEST 2 — with fields filter (current code)")
print("=" * 60)
resp2 = requests.get(f"{BASE}/schedule", params={
    "sportId":   1,
    "startDate": "2024-04-01",
    "endDate":   "2024-04-03",
    "gameType":  "R",
    "hydrate":   "linescore,decisions,probablePitcher(note)",
    "fields":    "dates,games,gamePk,gameDate,status,teams,score,probablePitcher,decisions,linescore",
}, timeout=15)
print("HTTP:", resp2.status_code)
data2 = resp2.json()
dates2 = data2.get("dates", [])
total2 = sum(len(d.get("games", [])) for d in dates2)
print("Total games with fields filter:", total2)

if total2 > 0:
    g2 = dates2[0]["games"][0]
    print("  status:", json.dumps(g2.get("status", {}), indent=4))
    home2 = g2.get("teams", {}).get("home", {})
    print("  home.score:", home2.get("score"))
    print("  home.isWinner:", home2.get("isWinner"))
else:
    print("  !! fields filter returned 0 games — this is the bug !!")
    print("  Raw response (first 500 chars):", str(data2)[:500])

print()
print("=" * 60)
print("TEST 3 — check abstractGameState values in real data")
print("=" * 60)
if total_games > 0:
    states = set()
    for d in dates:
        for g in d.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "MISSING")
            states.add(state)
    print("Unique abstractGameState values found:", states)
    print("(Code filters for 'Final' — if not in this set, 0 games result)")
