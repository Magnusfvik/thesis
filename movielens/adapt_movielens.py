#!/usr/bin/env python3
"""
adapt_movielens.py
==================
Converts MovieLens-1M .dat files into AMBAR-compatible CSVs so the
existing generate_complete_idun.py pipeline works unchanged.

Input  (thesis/movielens/):  ratings.dat, movies.dat
Output (thesis/movielens/data/):
    ratings_info.csv  — user_id, track_id, rating, timestamp
    tracks_info.csv   — track_id, styles, category_styles

Usage:
    python adapt_movielens.py
"""

import os
import pandas as pd

IN_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(IN_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Ratings
# ---------------------------------------------------------------------------
print("Converting ratings.dat ...")
ratings = pd.read_csv(
    os.path.join(IN_DIR, "ratings.dat"),
    sep="::",
    engine="python",
    names=["user_id", "track_id", "rating", "timestamp"],
)
ratings.to_csv(os.path.join(OUT_DIR, "ratings_info.csv"), index=False)
print(f"  {len(ratings):,} ratings | "
      f"{ratings['user_id'].nunique():,} users | "
      f"{ratings['track_id'].nunique():,} movies")

# ---------------------------------------------------------------------------
# 2. Movies → tracks_info
#    Genres are pipe-separated (same as AMBAR styles/category_styles)
# ---------------------------------------------------------------------------
print("Converting movies.dat ...")
movies = pd.read_csv(
    os.path.join(IN_DIR, "movies.dat"),
    sep="::",
    engine="python",
    names=["track_id", "title", "genres"],
    encoding="latin-1",
)
# Use genres as both styles and category_styles
# (MovieLens has no finer/coarser split — both columns are identical)
movies["styles"]          = movies["genres"]
movies["category_styles"] = movies["genres"]
movies[["track_id", "styles", "category_styles"]].to_csv(
    os.path.join(OUT_DIR, "tracks_info.csv"), index=False
)
print(f"  {len(movies):,} movies | genres sample: {movies['genres'].iloc[0]}")

print(f"\nDone. Files written to {OUT_DIR}/")
print("  ratings_info.csv")
print("  tracks_info.csv")
print("\nNext: run generate_complete_idun.py --data_dir movielens/data/")
