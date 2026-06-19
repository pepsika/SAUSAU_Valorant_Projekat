"""Ucitavanje sirovih CSV podataka i kanonizacija imena timova."""

import os

import pandas as pd

from config import ALIAS_MAPPING, GODINE, PATH


def kanonizuj_imena(df, kolone=None):
    """Zamijeni alias imena u zadatim kolonama."""
    if kolone is None:
        kolone = ["Team A", "Team B", "Team", "Teams"]
    for kol in kolone:
        if kol in df.columns:
            df[kol] = df[kol].replace(ALIAS_MAPPING)
    return df


def ucitaj_csv_po_godinama(relativna_putanja, naziv=""):
    """Ucita isti CSV iz svih godina i spoji ih, dodajuci kolonu 'godina'."""
    frames = []
    for godina in GODINE:
        fp = os.path.join(PATH, "2021-2026", godina, relativna_putanja)
        if os.path.exists(fp):
            df = pd.read_csv(fp, low_memory=False)
            df["godina"] = godina
            frames.append(df)
    df_spojen = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if naziv:
        print(f"  {naziv}: {len(df_spojen)} redova")
    return df_spojen


def cisto_procenti(serija):
    """Pretvara '75%' u 0.75; nevazece u NaN."""
    return pd.to_numeric(
        serija.astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    ) / 100.0


def ucitaj_stage1(path=PATH):
    """Ucita Stage 1 2026 standings i mecevi po regionima (kvalifikacije za London)."""
    stage1_path = os.path.join(path, "2021-2026", "vct_2026", "stage 1",
                                "stage1_final_standings_all_regions.csv")
    df_stage1 = pd.read_csv(stage1_path)
    print(f"  stage1 standings: {len(df_stage1)} timova")

    stage1_mecevi = []
    for region in ["americas", "emea", "pacific", "china"]:
        fp = os.path.join(path, "2021-2026", "vct_2026", "stage 1", region,
                           "matches", "match_scores.csv")
        if os.path.exists(fp):
            df_r = pd.read_csv(fp)
            df_r["region"] = region
            stage1_mecevi.append(df_r)
    df_stage1_mecevi = pd.concat(stage1_mecevi, ignore_index=True) if stage1_mecevi else pd.DataFrame()
    print(f"  stage1 mecevi: {len(df_stage1_mecevi)} meceva")
    return df_stage1, df_stage1_mecevi


def load_all_data():
    """Ucita sve potrebne CSV-ove, kanonizuje imena timova i vrati ih kao dict.

    Kanonizacija je KRITICNA - inace npr. NRG izgleda kao slab tim jer su
    "Mega Minors" mecevi (NRG pod drugim imenom) razdvojeni od pravih NRG meceva.
    """
    print("=" * 60)
    print("  UCITAVANJE PODATAKA")
    print("=" * 60)

    df_pro = pd.read_csv(os.path.join(PATH, "dataset_sa_featurima.csv"))
    print(f"Osnovni dataset: {len(df_pro)} meceva")

    print("\nUcitavam dodatne fajlove:")
    df_eco = ucitaj_csv_po_godinama("matches/eco_stats.csv", "eco_stats")
    df_overview = ucitaj_csv_po_godinama("matches/overview.csv", "overview")
    df_players = ucitaj_csv_po_godinama("players_stats/players_stats.csv", "players")
    df_maps_scores = ucitaj_csv_po_godinama("matches/maps_scores.csv", "maps_scores")
    df_teams_agents = ucitaj_csv_po_godinama("agents/teams_picked_agents.csv", "teams_agents")

    df_stage1, df_stage1_mecevi = ucitaj_stage1()

    print("\n  Kanonizujem aliase timova...")
    for df_x in [df_pro, df_eco, df_overview, df_players, df_maps_scores, df_teams_agents]:
        kanonizuj_imena(df_x)
    df_stage1["team"] = df_stage1["team"].replace(ALIAS_MAPPING)
    if not df_stage1_mecevi.empty:
        df_stage1_mecevi["team_a"] = df_stage1_mecevi["team_a"].replace(ALIAS_MAPPING)
        df_stage1_mecevi["team_b"] = df_stage1_mecevi["team_b"].replace(ALIAS_MAPPING)
        df_stage1_mecevi["winner"] = df_stage1_mecevi["winner"].replace(ALIAS_MAPPING)

    provj = df_pro[((df_pro["Team A"] == "NRG") | (df_pro["Team B"] == "NRG")) &
                   (df_pro["Tournament"].str.contains("Champions 2025|Santiago 2026|Toronto 2025", na=False))]
    print(f"  Provjera: NRG mecevi na tier1 (Champions 2025/Santiago 2026/Toronto 2025): {len(provj)}")

    return {
        "df_pro": df_pro,
        "df_eco": df_eco,
        "df_overview": df_overview,
        "df_players": df_players,
        "df_maps_scores": df_maps_scores,
        "df_teams_agents": df_teams_agents,
        "df_stage1": df_stage1,
        "df_stage1_mecevi": df_stage1_mecevi,
    }
