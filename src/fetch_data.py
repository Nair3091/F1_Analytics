"""
fetch_data.py
─────────────
Handles all data ingestion:
  - Verifies Ergast CSV files (manually downloaded from Kaggle)
  - Sequentially fetches OpenF1 API data (2023–2025)

Usage:
    python src/fetch_data.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import logging
import aiohttp

from config import (
    OPENF1_BASE_URL, OPENF1_YEARS, OPENF1_ENDPOINTS,
    ERGAST_FILES,
    OPENF1_RAW, ERGAST_RAW,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(OPENF1_RAW, exist_ok=True)
    os.makedirs(ERGAST_RAW, exist_ok=True)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def session_already_fetched(session_dir):
    """
    Returns True if all endpoints were already fetched and have data.
    Skips re-fetching sessions that completed successfully before.
    """
    if not os.path.exists(session_dir):
        return False
    for ep in OPENF1_ENDPOINTS:
        path = os.path.join(session_dir, f"{ep}.json")
        if not os.path.exists(path):
            return False
    return True


# ── Ergast ─────────────────────────────────────────────────────────────────

def verify_ergast():
    """
    Checks that all required Ergast CSV files exist.
    Data must be manually downloaded from Kaggle and placed in data/raw/ergast/
    https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
    """
    missing = [f for f in ERGAST_FILES
               if not os.path.exists(os.path.join(ERGAST_RAW, f))]

    if missing:
        log.error("Missing Ergast CSV files. Please download manually from Kaggle:")
        log.error("https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
        log.error(f"Missing files: {missing}")
        raise FileNotFoundError(f"Missing Ergast CSVs: {missing}")

    log.info(f"All Ergast CSV files found in {ERGAST_RAW}")


# ── OpenF1 ─────────────────────────────────────────────────────────────────

async def fetch_endpoint(session, endpoint, params, retries=5):
    """
    Fetch a single OpenF1 endpoint with exponential backoff retry.
    Returns parsed JSON list or empty list on failure.
    """
    url = f"{OPENF1_BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    wait = 2 ** attempt * 3   # 3s, 6s, 12s, 24s, 48s
                    log.warning(f"Rate limited on {endpoint}, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                elif resp.status == 404:
                    log.warning(f"HTTP 404 on {endpoint} params={params}")
                    return []
                else:
                    log.warning(f"HTTP {resp.status} on {endpoint} params={params}")
                    return []
        except Exception as e:
            wait = 2 ** attempt * 2
            log.error(f"Error fetching {endpoint} (attempt {attempt+1}): {e}, retrying in {wait}s...")
            await asyncio.sleep(wait)
    log.error(f"Gave up on {endpoint} after {retries} attempts.")
    return []


async def fetch_race_sessions(year, retries=5):
    """
    Returns a list of race session dicts for a given year.
    Uses longer retries since this is the critical first call per year.
    """
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            data = await fetch_endpoint(
                session, "sessions",
                {"year": year, "session_type": "Race"},
                retries=1
            )
            if data:
                return data
            wait = 2 ** attempt * 5   # 5s, 10s, 20s, 40s, 80s
            log.warning(f"No sessions returned for {year}, retrying in {wait}s...")
            await asyncio.sleep(wait)
    return []


async def fetch_session_data(session_key, year):
    """
    Fetches all endpoints for a single race session SEQUENTIALLY
    to avoid hammering the rate limit.
    Skips sessions that were already fully fetched.
    """
    session_dir = os.path.join(OPENF1_RAW, str(year), str(session_key))

    if session_already_fetched(session_dir):
        log.info(f"  Already fetched session {session_key}, skipping.")
        return

    os.makedirs(session_dir, exist_ok=True)

    async with aiohttp.ClientSession() as http_session:
        for ep in OPENF1_ENDPOINTS:
            out_path = os.path.join(session_dir, f"{ep}.json")

            # Skip individual endpoints that already exist
            if os.path.exists(out_path):
                log.info(f"  {ep}.json already exists, skipping.")
                continue

            result = await fetch_endpoint(http_session, ep, {"session_key": session_key})

            if isinstance(result, list):
                save_json(result, out_path)
                log.info(f"  Saved {ep}.json ({len(result)} records)")
            else:
                log.error(f"  Failed to fetch {ep}")
                save_json([], out_path)   # save empty so we don't retry forever

            await asyncio.sleep(2)   # 2s between each endpoint call


async def fetch_openf1_year(year):
    """
    Fetches all race session data for a given year.
    """
    log.info(f"Fetching OpenF1 data for {year}...")
    sessions = await fetch_race_sessions(year)

    if not sessions:
        log.warning(f"No race sessions found for {year} after all retries.")
        return

    log.info(f"Found {len(sessions)} race sessions for {year}.")

    for s in sessions:
        session_key = s.get("session_key")
        name = s.get("meeting_name", s.get("location", "unknown"))

        # Save session metadata
        session_dir = os.path.join(OPENF1_RAW, str(year), str(session_key))
        os.makedirs(session_dir, exist_ok=True)
        meta_path = os.path.join(session_dir, "session_meta.json")
        if not os.path.exists(meta_path):
            save_json(s, meta_path)

        log.info(f"  Fetching: {name} (session_key={session_key})")
        await fetch_session_data(session_key, year)

        # Pause between sessions to respect rate limits
        log.info(f"  Pausing 5s before next session...")
        await asyncio.sleep(5)


async def fetch_all_openf1():
    """
    Fetches OpenF1 data for all configured years sequentially.
    Pauses between years to let the rate limit window reset.
    """
    for year in OPENF1_YEARS:
        await fetch_openf1_year(year)
        log.info(f"Finished {year}. Pausing 30s before next year...")
        await asyncio.sleep(30)
    log.info("OpenF1 race session ingestion complete.")


# ── Qualifying sessions (OpenF1) ───────────────────────────────────────────

async def fetch_qualifying_sessions():
    """
    Fetches qualifying session lap data (needed for grid position features).
    """
    log.info("Fetching qualifying session data from OpenF1...")

    async with aiohttp.ClientSession() as http_session:
        for year in OPENF1_YEARS:
            log.info(f"  Qualifying sessions for {year}...")
            sessions = await fetch_endpoint(
                http_session, "sessions",
                {"year": year, "session_type": "Qualifying"}
            )

            if not sessions:
                log.warning(f"  No qualifying sessions found for {year}.")
                await asyncio.sleep(15)
                continue

            for s in sessions:
                session_key = s.get("session_key")
                name = s.get("meeting_name", s.get("location", "unknown"))
                out_dir = os.path.join(OPENF1_RAW, str(year), f"quali_{session_key}")
                out_path = os.path.join(out_dir, "laps.json")

                if os.path.exists(out_path):
                    log.info(f"  Already fetched quali laps for {name}, skipping.")
                    continue

                os.makedirs(out_dir, exist_ok=True)
                log.info(f"  Qualifying: {name} ({year})")

                laps = await fetch_endpoint(
                    http_session, "laps",
                    {"session_key": session_key}
                )
                save_json(laps, out_path)
                log.info(f"    Saved laps.json ({len(laps)} records)")

                await asyncio.sleep(3)   # 3s between qualifying sessions

            log.info(f"  Done with {year} qualifying. Pausing 20s...")
            await asyncio.sleep(20)


# ── Entry point ────────────────────────────────────────────────────────────

def run():
    ensure_dirs()

    # Step 1: Verify Ergast CSV files
    log.info("=== Step 1: Ergast (1950–2022) ===")
    verify_ergast()

    # Step 2: OpenF1 race session data
    log.info("=== Step 2: OpenF1 Race Sessions (2023–2025) ===")
    asyncio.run(fetch_all_openf1())

    # Step 3: OpenF1 qualifying lap data
    log.info("=== Step 3: OpenF1 Qualifying Laps (2023–2025) ===")
    asyncio.run(fetch_qualifying_sessions())

    log.info("=== All ingestion complete ===")


if __name__ == "__main__":
    run()