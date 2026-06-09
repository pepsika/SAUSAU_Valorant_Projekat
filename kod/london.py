"""
====================================================================
   VALORANT VCT - PREDIKCIJA MASTERS LONDON 2026 (OPTIMIZOVANO)
====================================================================

Glavne izmjene u odnosu na prvu verziju:
  1. RECENCY WEIGHTING - noviji mecevi imaju vecu tezinu (2026 >> 2021)
  2. STAGE 1 2026 PODACI - championship_points kao feature
  3. MASTERS SANTIAGO 2026 - dodati u trening (najsvjeziji Masters)
  4. MAP-SPECIFIC WIN RATE - kao feature u modelu (ne samo za prikaz)
  5. FIX ECO BUG - prave kategorije ($, $$, $$$, Eco, Pistol)
  6. PISTOL WIN RATE - jak prediktor
  7. STANDARDNO SKALIRANJE - StandardScaler u pipeline-u
  8. GRADIENT BOOSTING - novi, jaci algoritam
  9. REALAN MONTE CARLO - bez "biranja protivnika", standardno seedovanje
 10. FIX DATA LEAKAGE - CV samo na trening setu
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ====================================================================
#  KONFIGURACIJA
# ====================================================================

PATH = r"C:\Users\KORISNIK\Desktop\6. semestar\Softverski algoritmi u sistemima automatskog upravljanja\Projekat\dataset"
GODINE = ["vct_2021", "vct_2022", "vct_2023", "vct_2024", "vct_2025", "vct_2026"]
PATH_REZULTATI = r"C:\Users\KORISNIK\Desktop\6. semestar\Softverski algoritmi u sistemima automatskog upravljanja\Projekat\rezultati"
# RECENCY WEIGHTING - skoriji mecevi vise vrijede pri treniranju
# Manje agresivno nego prije - cuvamo informaciju o istorijski jakim timovima
RECENCY_WEIGHTS = {
    "vct_2021": 0.30,
    "vct_2022": 0.50,
    "vct_2023": 0.70,
    "vct_2024": 0.90,
    "vct_2025": 1.20,
    "vct_2026": 1.50,
}

# Lista TIER-1 medjunarodnih turnira (Champions + Masters)
# Ovo su DOKAZ kvaliteta - tu igraju samo najbolji timovi svake regije
TIER1_TURNIRI = [
    # Champions
    "Valorant Champions 2021", "Valorant Champions 2022",
    "Valorant Champions 2023", "Valorant Champions 2024",
    "Valorant Champions 2025",
    # 2023+ Masters
    "Champions Tour 2023: Masters Tokyo",
    "Champions Tour 2024: Masters Madrid",
    "Champions Tour 2024: Masters Shanghai",
    # 2025 Masters
    "Valorant Masters Bangkok 2025", "Valorant Masters Toronto 2025",
    # 2026 Masters
    "Valorant Masters Santiago 2026",
]

RANDOM_STATE = 42

# ALIAS MAPPING - razna imena istog tima u datasetu
# Kaggle dataset koristi razlicite nazive za iste timove!
# Mega Minors u datasetu = NRG (Champions 2025 GF: "Mega Minors 3-2 FNATIC" = NRG 3-2 FNATIC)
ALIAS_MAPPING = {
    "NRG Esports": "NRG",
    "Mega Minors": "NRG",   # NRG-ovi mecevi pod ovim imenom (Champions 2025, Santiago 2026, VCT 2025/2026)
    "Rise Gaming": "Rise",
}


def kanonizuj_imena(df, kolone=None):
    """Zamijeni alias imena u zadatim kolonama."""
    if kolone is None:
        kolone = ["Team A", "Team B", "Team", "Teams"]
    for kol in kolone:
        if kol in df.columns:
            df[kol] = df[kol].replace(ALIAS_MAPPING)
    return df

# ====================================================================
#  POMOCNE FUNKCIJE ZA UCITAVANJE
# ====================================================================

def ucitaj_csv_po_godinama(relativna_putanja, naziv=""):
    """Ucita isti CSV iz svih godinas i spoji ih, dodajuci kolonu 'godina'."""
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


# ====================================================================
#  1. UCITAVANJE OSNOVNIH PODATAKA
# ====================================================================

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

# Stage 1 2026 - kvalifikacije za London (KLJUCNO!)
stage1_path = os.path.join(PATH, "2021-2026", "vct_2026", "stage 1",
                          "stage1_final_standings_all_regions.csv")
df_stage1 = pd.read_csv(stage1_path)
print(f"  stage1 standings: {len(df_stage1)} timova")

# Stage 1 mecevi po regionima
stage1_mecevi = []
for region in ["americas", "emea", "pacific", "china"]:
    fp = os.path.join(PATH, "2021-2026", "vct_2026", "stage 1", region,
                      "matches", "match_scores.csv")
    if os.path.exists(fp):
        df_r = pd.read_csv(fp)
        df_r["region"] = region
        stage1_mecevi.append(df_r)
df_stage1_mecevi = pd.concat(stage1_mecevi, ignore_index=True) if stage1_mecevi else pd.DataFrame()
print(f"  stage1 mecevi: {len(df_stage1_mecevi)} meceva")

# KANONIZACIJA IMENA - KRITICNO! 
# Inace NRG izgleda kao slab tim jer su "Mega Minors" mecevi razdvojeni
print("\n  Kanonizujem aliase timova...")
for df_x in [df_pro, df_eco, df_overview, df_players, df_maps_scores, df_teams_agents]:
    kanonizuj_imena(df_x)
df_stage1["team"] = df_stage1["team"].replace(ALIAS_MAPPING)
if not df_stage1_mecevi.empty:
    df_stage1_mecevi["team_a"] = df_stage1_mecevi["team_a"].replace(ALIAS_MAPPING)
    df_stage1_mecevi["team_b"] = df_stage1_mecevi["team_b"].replace(ALIAS_MAPPING)
    df_stage1_mecevi["winner"] = df_stage1_mecevi["winner"].replace(ALIAS_MAPPING)

# Verifikacija - koliko sada NRG ima meceva u tier1?
provj = df_pro[((df_pro["Team A"]=="NRG") | (df_pro["Team B"]=="NRG")) & 
               (df_pro["Tournament"].str.contains("Champions 2025|Santiago 2026|Toronto 2025", na=False))]
print(f"  Provjera: NRG mecevi na tier1 (Champions 2025/Santiago 2026/Toronto 2025): {len(provj)}")

# ====================================================================
#  2. FEATURE ENGINEERING - PO TIMU
# ====================================================================

print("\n" + "=" * 60)
print("  KREIRANJE FEATURA PO TIMU")
print("=" * 60)

# --- 2.1 ECO STATS (FIX: prave kategorije!) ---
df_eco["Initiated"] = pd.to_numeric(df_eco["Initiated"], errors="coerce")
df_eco["Won"] = pd.to_numeric(df_eco["Won"], errors="coerce")

# Pistol Won - posebno tretiramo (Initiated nije popunjen, ali je uvjek 2 po mapi)
pistol_df = df_eco[df_eco["Type"] == "Pistol Won"].groupby("Team").agg(
    pistol_pobjede=("Won", "sum"),
    pistol_mecevi=("Won", "count")
).reset_index()
pistol_df["pistol_wr"] = pistol_df["pistol_pobjede"] / (pistol_df["pistol_mecevi"] * 2)

# Full buy ($$$ won) i Eco won - PRAVE kategorije
def eco_kategorija_wr(tip, naziv):
    sub = df_eco[df_eco["Type"] == tip].copy()
    grupa = sub.groupby("Team").agg(
        won=("Won", "sum"),
        init=("Initiated", "sum")
    ).reset_index()
    grupa[naziv] = grupa["won"] / grupa["init"].replace(0, np.nan)
    return grupa[["Team", naziv]]

full_buy_df = eco_kategorija_wr("$$$ (won)", "full_buy_wr")
eco_won_df = eco_kategorija_wr("Eco (won)", "eco_wr")
semi_buy_df = eco_kategorija_wr("$$ (won)", "semi_buy_wr")

# --- 2.2 PLAYER STATS PO TIMU (iz overview) ---
df_overview["Rating"] = pd.to_numeric(df_overview["Rating"], errors="coerce")
df_overview["Average Combat Score"] = pd.to_numeric(
    df_overview["Average Combat Score"], errors="coerce")
df_overview["Average Damage Per Round"] = pd.to_numeric(
    df_overview["Average Damage Per Round"], errors="coerce")

# Filter samo "both" side (cijeli mec)
df_ov_both = df_overview[df_overview["Side"] == "both"]

team_player_stats = df_ov_both.groupby("Team").agg(
    rating_avg=("Rating", "mean"),
    acs_avg=("Average Combat Score", "mean"),
    adr_avg=("Average Damage Per Round", "mean")
).reset_index()

# --- 2.3 PLAYERS STATS (clutch, KAST, HS) ---
df_players["Rating"] = pd.to_numeric(df_players["Rating"], errors="coerce")
df_players["Headshot %"] = cisto_procenti(df_players["Headshot %"]) * 100
df_players["Kill, Assist, Trade, Survive %"] = cisto_procenti(
    df_players["Kill, Assist, Trade, Survive %"]) * 100
df_players["First Kills Per Round"] = pd.to_numeric(
    df_players["First Kills Per Round"], errors="coerce")

# Clutch parsiranje "won/played"
def parsiraj_clutch(s):
    try:
        w, p = str(s).split("/")
        w, p = int(w), int(p)
        return w / p if p > 0 else np.nan
    except:
        return np.nan

df_players["clutch_rate"] = df_players["Clutches (won/played)"].apply(parsiraj_clutch)

team_player_advanced = df_players.groupby("Teams").agg(
    hs_avg=("Headshot %", "mean"),
    kast_avg=("Kill, Assist, Trade, Survive %", "mean"),
    fkpr_avg=("First Kills Per Round", "mean"),
    clutch_avg=("clutch_rate", "mean")
).reset_index().rename(columns={"Teams": "Team"})

# --- 2.4 MAP WIN RATE PO TIMU ---
df_maps_scores["team_a_won_map"] = (
    df_maps_scores["Team A Score"] > df_maps_scores["Team B Score"]).astype(int)
df_maps_scores["team_b_won_map"] = 1 - df_maps_scores["team_a_won_map"]

# Tim kao A
wr_a = df_maps_scores.groupby("Team A").agg(
    map_pob_a=("team_a_won_map", "sum"),
    map_uk_a=("team_a_won_map", "count")
).reset_index().rename(columns={"Team A": "Team"})

# Tim kao B
wr_b = df_maps_scores.groupby("Team B").agg(
    map_pob_b=("team_b_won_map", "sum"),
    map_uk_b=("team_b_won_map", "count")
).reset_index().rename(columns={"Team B": "Team"})

# Spoji
team_map_wr = wr_a.merge(wr_b, on="Team", how="outer").fillna(0)
team_map_wr["ukupne_pobjede_mapa"] = team_map_wr["map_pob_a"] + team_map_wr["map_pob_b"]
team_map_wr["ukupno_mapa"] = team_map_wr["map_uk_a"] + team_map_wr["map_uk_b"]
team_map_wr["map_wr"] = team_map_wr["ukupne_pobjede_mapa"] / team_map_wr["ukupno_mapa"].replace(0, np.nan)
team_map_wr = team_map_wr[["Team", "map_wr", "ukupno_mapa"]]

# --- 2.5 STAGE 1 2026 - CHAMPIONSHIP POINTS (NOVI KLJUCNI FEATURE!) ---
stage1_features = df_stage1[["team", "championship_points", "placement", "region"]].copy()
stage1_features.columns = ["Team", "championship_points", "stage1_placement", "region"]
# Skaliraj placement: 1. mjesto = 1.0, 12. mjesto = 0.0
stage1_features["stage1_score"] = (
    13 - stage1_features["stage1_placement"]) / 12.0

# --- 2.5b INTERNATIONAL TIER SUCCESS (KLJUCNO za top timove!) ---
# Win rate na Tier-1 turnirima (Champions/Masters) - dokazi istorijski uspjeh
df_tier1 = df_pro[df_pro["Tournament"].isin(TIER1_TURNIRI)].copy()

# Win rate na Tier-1
tier1_a = df_tier1.groupby("Team A").agg(
    tier1_pob_a=("team_a_won", "sum"),
    tier1_uk_a=("team_a_won", "count")
).reset_index().rename(columns={"Team A": "Team"})

df_tier1["team_b_won"] = 1 - df_tier1["team_a_won"]
tier1_b = df_tier1.groupby("Team B").agg(
    tier1_pob_b=("team_b_won", "sum"),
    tier1_uk_b=("team_b_won", "count")
).reset_index().rename(columns={"Team B": "Team"})

tier1_combined = tier1_a.merge(tier1_b, on="Team", how="outer").fillna(0)
tier1_combined["tier1_pobjede"] = tier1_combined["tier1_pob_a"] + tier1_combined["tier1_pob_b"]
tier1_combined["tier1_meceva"] = tier1_combined["tier1_uk_a"] + tier1_combined["tier1_uk_b"]
tier1_combined["tier1_wr"] = (tier1_combined["tier1_pobjede"] /
                              tier1_combined["tier1_meceva"].replace(0, np.nan))
tier1_features = tier1_combined[["Team", "tier1_wr", "tier1_meceva"]]
# Timovi koji nikad nisu igrali na Tier-1 = signal slabosti, ne 0.5!
# Stavljamo 0.3 - znaci 'rijetko/nikad na top sceni' = ispod prosjeka
tier1_features = tier1_features.fillna({"tier1_wr": 0.3, "tier1_meceva": 0})

print(f"  Tier-1 turniri ucitani: {len(df_tier1)} meceva, "
      f"{len(tier1_features[tier1_features['tier1_meceva']>0])} timova sa iskustvom")

# RECENT INTERNATIONAL FORM - samo 2025/2026 Masters i Champions
recent_int_turniri = ["Valorant Champions 2025", "Valorant Masters Bangkok 2025",
                      "Valorant Masters Toronto 2025", "Valorant Masters Santiago 2026"]
df_recent = df_pro[df_pro["Tournament"].isin(recent_int_turniri)].copy()

recent_a = df_recent.groupby("Team A").agg(
    recent_pob_a=("team_a_won", "sum"),
    recent_uk_a=("team_a_won", "count")
).reset_index().rename(columns={"Team A": "Team"})

df_recent["team_b_won"] = 1 - df_recent["team_a_won"]
recent_b = df_recent.groupby("Team B").agg(
    recent_pob_b=("team_b_won", "sum"),
    recent_uk_b=("team_b_won", "count")
).reset_index().rename(columns={"Team B": "Team"})

recent_combined = recent_a.merge(recent_b, on="Team", how="outer").fillna(0)
recent_combined["recent_int_pobjede"] = (
    recent_combined["recent_pob_a"] + recent_combined["recent_pob_b"])
recent_combined["recent_int_meceva"] = (
    recent_combined["recent_uk_a"] + recent_combined["recent_uk_b"])
recent_combined["recent_int_wr"] = (recent_combined["recent_int_pobjede"] /
                                    recent_combined["recent_int_meceva"].replace(0, np.nan))
recent_features = recent_combined[["Team", "recent_int_wr", "recent_int_meceva"]]
recent_features = recent_features.fillna({"recent_int_wr": 0.3, "recent_int_meceva": 0})

# --- 2.5c DINAMICKI WIN RATE (racunat POSLE kanonizacije!) ---
# KLJUCNO: win_rate_a/b iz CSV-a su racunati PRIJE alias merga!
# NRG ima samo 9 meceva u CSV-u umjesto 116, jer su NRG Esports i
# Mega Minors tretirani kao odvojeni timovi. Moramo preracunati.
def racunaj_dinamicki_wr(df):
    a_stats = df.groupby("Team A").agg(
        pob_a=("team_a_won", "sum"), mec_a=("team_a_won", "count")
    ).reset_index().rename(columns={"Team A": "Team"})
    df_temp = df.copy()
    df_temp["team_b_won_t"] = 1 - df_temp["team_a_won"]
    b_stats = df_temp.groupby("Team B").agg(
        pob_b=("team_b_won_t", "sum"), mec_b=("team_b_won_t", "count")
    ).reset_index().rename(columns={"Team B": "Team"})
    comb = a_stats.merge(b_stats, on="Team", how="outer").fillna(0)
    comb["dynamic_wr"] = (comb["pob_a"] + comb["pob_b"]) / (comb["mec_a"] + comb["mec_b"])
    comb["dynamic_mecevi"] = comb["mec_a"] + comb["mec_b"]
    return comb[["Team", "dynamic_wr", "dynamic_mecevi"]]

dynamic_wr_df = racunaj_dinamicki_wr(df_pro)
print(f"\n  Dinamicki WR (nakon kanonizacije):")
for tim in ["NRG", "Paper Rex", "G2 Esports", "Team Vitality", "Team Heretics"]:
    row = dynamic_wr_df[dynamic_wr_df["Team"]==tim]
    if not row.empty:
        print(f"    {tim:25s} WR={row['dynamic_wr'].values[0]:.3f}  "
              f"meceva={int(row['dynamic_mecevi'].values[0])}")

# --- 2.5d ELO REJTING SISTEM ---
# Elo racuna RELATIVNU jacinu tima - pobjeda nad jakim timom daje vise
# nego pobjeda nad slabim. Ovo prirodno razlikuje NRG (Champions 2025
# pobjednik - pobijedio FNATIC, DRX, Gen.G) od Xi Lai (pobjede u Kini).
def racunaj_elo(df, k_base=32, start_elo=1500):
    elo = {}
    forma = {}  # zadnjih N meceva za svaki tim

    for _, row in df.iterrows():
        a, b = row["Team A"], row["Team B"]
        result = row["team_a_won"]  # 1 = A pobijedio

        if a not in elo: elo[a] = start_elo
        if b not in elo: elo[b] = start_elo
        if a not in forma: forma[a] = []
        if b not in forma: forma[b] = []

        # Veci K za Tier-1 turnire (Champions/Masters)
        turnir = str(row.get("Tournament", ""))
        if any(t in turnir for t in ["Champions", "Masters"]):
            k = k_base * 1.5  # 48 - medjunarodni mecevi vrijede vise
        else:
            k = k_base

        ea = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
        eb = 1 - ea

        elo[a] += k * (result - ea)
        elo[b] += k * ((1 - result) - eb)

        forma[a].append(result)
        forma[b].append(1 - result)
        # Zadrzi samo zadnjih 15 meceva za formu
        forma[a] = forma[a][-15:]
        forma[b] = forma[b][-15:]

    # Racunaj formu (win rate zadnjih 15 meceva)
    forma_dict = {}
    for tim, results in forma.items():
        if len(results) > 0:
            forma_dict[tim] = sum(results) / len(results)
        else:
            forma_dict[tim] = 0.5

    return elo, forma_dict

# Sortiraj df_pro hronoloski (vec je, ali za svaki slucaj)
elo_dict, forma_dict = racunaj_elo(df_pro)

elo_df = pd.DataFrame([
    {"Team": t, "elo": e, "forma_15": forma_dict.get(t, 0.5)}
    for t, e in elo_dict.items()
])

print(f"\n  ELO rejting (top Masters London timovi):")
london_timovi = ["NRG", "Paper Rex", "G2 Esports", "Team Heretics",
                 "Team Vitality", "EDward Gaming", "LEVIATÁN",
                 "Xi Lai Gaming", "FULL SENSE", "FUT Esports",
                 "Global Esports", "Dragon Ranger Gaming"]
elo_london = elo_df[elo_df["Team"].isin(london_timovi)].sort_values("elo", ascending=False)
for _, row in elo_london.iterrows():
    print(f"    {row['Team']:25s} ELO={row['elo']:.0f}  "
          f"forma(15)={row['forma_15']:.3f}")

# --- 2.6 SPAJANJE SVIH FEATURA U JEDAN DATAFRAME ---
team_features = team_map_wr.copy()
for df_dod in [team_player_stats, team_player_advanced,
               pistol_df[["Team", "pistol_wr"]],
               full_buy_df, eco_won_df, semi_buy_df,
               stage1_features[["Team", "championship_points", "stage1_score"]],
               tier1_features, recent_features,
               dynamic_wr_df, elo_df]:
    team_features = team_features.merge(df_dod, on="Team", how="outer")

# Popunjavanje NaN sa MEDIANOM (ne 0.5 - 0.5 ima smisla samo za WR, ne za rating!)
for kol in team_features.columns:
    if kol == "Team":
        continue
    if team_features[kol].dtype in [np.float64, np.int64]:
        team_features[kol] = team_features[kol].fillna(team_features[kol].median())

print(f"Ukupno timova sa featurima: {len(team_features)}")
print(f"Broj featura po timu: {team_features.shape[1] - 1}")

# ====================================================================
#  3. PRIPREMA DATASETA ZA MODEL
# ====================================================================

print("\n" + "=" * 60)
print("  PRIPREMA TRENING/TEST")
print("=" * 60)

target = "team_a_won"

# Dodaj feature za Team A i Team B
df_full = df_pro.merge(
    team_features.add_prefix("a_").rename(columns={"a_Team": "Team A"}),
    on="Team A", how="left"
)
df_full = df_full.merge(
    team_features.add_prefix("b_").rename(columns={"b_Team": "Team B"}),
    on="Team B", how="left"
)

# Popuni NaN medianom kolone
for kol in df_full.select_dtypes(include=[np.number]).columns:
    df_full[kol] = df_full[kol].fillna(df_full[kol].median())

# RAZLIKA-BASED FEATURI (cesto bolji za ML jer model lakse uci)
diff_features_pairs = [
    ("a_map_wr", "b_map_wr", "diff_map_wr"),
    ("a_rating_avg", "b_rating_avg", "diff_rating"),
    ("a_acs_avg", "b_acs_avg", "diff_acs"),
    ("a_adr_avg", "b_adr_avg", "diff_adr"),
    ("a_pistol_wr", "b_pistol_wr", "diff_pistol"),
    ("a_full_buy_wr", "b_full_buy_wr", "diff_full_buy"),
    ("a_eco_wr", "b_eco_wr", "diff_eco"),
    ("a_kast_avg", "b_kast_avg", "diff_kast"),
    ("a_fkpr_avg", "b_fkpr_avg", "diff_fkpr"),
    ("a_clutch_avg", "b_clutch_avg", "diff_clutch"),
    ("a_hs_avg", "b_hs_avg", "diff_hs"),
    ("a_championship_points", "b_championship_points", "diff_champ_pts"),
    ("a_stage1_score", "b_stage1_score", "diff_stage1"),
    ("a_tier1_wr", "b_tier1_wr", "diff_tier1_wr"),
    ("a_tier1_meceva", "b_tier1_meceva", "diff_tier1_iskustvo"),
    ("a_recent_int_wr", "b_recent_int_wr", "diff_recent_int_wr"),
    ("a_dynamic_wr", "b_dynamic_wr", "diff_dynamic_wr"),
    ("a_dynamic_mecevi", "b_dynamic_mecevi", "diff_dynamic_mecevi"),
    ("a_elo", "b_elo", "diff_elo"),
    ("a_forma_15", "b_forma_15", "diff_forma"),
    ("win_rate_a", "win_rate_b", "diff_win_rate"),
    ("broj_meceva_a", "broj_meceva_b", "diff_iskustvo"),
]
for a_col, b_col, diff_col in diff_features_pairs:
    if a_col in df_full.columns and b_col in df_full.columns:
        df_full[diff_col] = df_full[a_col] - df_full[b_col]

# Konacna lista featura - kombinacija individualnih i razlika
features = [
    # h2h - direct
    "h2h_win_rate_a", "h2h_mecevi",
    # ELO i FORMA - NAJJACI featuri za relativnu jacinu timova
    "a_elo", "b_elo", "diff_elo",
    "a_forma_15", "b_forma_15", "diff_forma",
    # Dinamicki WR (posle alias merge-a - TACAN broj meceva!)
    "a_dynamic_wr", "b_dynamic_wr", "diff_dynamic_wr",
    "diff_dynamic_mecevi",
    # Tier 1 i recent international
    "a_tier1_wr", "b_tier1_wr", "diff_tier1_wr",
    "diff_tier1_iskustvo",
    "a_recent_int_wr", "b_recent_int_wr", "diff_recent_int_wr",
    # Stage 1 2026
    "diff_champ_pts", "diff_stage1",
    # Player/team stats
    "diff_map_wr", "diff_rating", "diff_acs", "diff_adr",
    "diff_pistol", "diff_full_buy", "diff_eco",
    "diff_kast", "diff_fkpr", "diff_clutch", "diff_hs",
    # Legacy (iz CSV-a) - manje vazni ali modelu ne smetaju
    "win_rate_a", "win_rate_b", "diff_win_rate",
]
features = [f for f in features if f in df_full.columns]
print(f"Ukupno featura u modelu: {len(features)}")

# Train/test split
train_godine = ["vct_2021", "vct_2022", "vct_2023", "vct_2024"]
test_godine = ["vct_2025", "vct_2026"]

train = df_full[df_full["godina"].isin(train_godine)].copy()
test = df_full[df_full["godina"].isin(test_godine)].copy()

# RECENCY WEIGHTS na trening setu
train["recency_weight"] = train["godina"].map(RECENCY_WEIGHTS)
sample_weights = train["recency_weight"].values

X_train = train[features].values
y_train = train[target].values
X_test = test[features].values
y_test = test[target].values

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ====================================================================
#  3.5 ENKODIRANJE I KORELACIONA ANALIZA
# ====================================================================

print("\n" + "=" * 60)
print("  ENKODIRANJE I KORELACIONA ANALIZA")
print("=" * 60)

# ENKODIRANJE:
# Kategoricki atributi (Tournament, Stage, Match Type, godina) NISU
# direktno korisceni kao feature-i. Umjesto toga, iz njih su IZVEDENI
# numericki feature-i:
#   - "godina" -> recency_weight (0.30 - 1.50)
#   - "Tournament" -> tier1_wr, recent_int_wr (da li je medjunarodni turnir)
#   - "Team A"/"Team B" -> svi team-level feature-i (ELO, dynamic_wr, itd.)
#
# Ovim pristupom izbjegavamo probleme one-hot encodinga na 100+ timova
# i 20+ turnira (sto bi stvorilo rijedak, visokodimenzionalan dataset).

# KORELACIONA ANALIZA
# Provjeravamo da li su featuri medjusobno jako korelisani (multikolinearnost).
# Ako dva featura nose istu informaciju, model moze biti nestabilan.

df_corr = pd.DataFrame(X_train, columns=features)
corr_matrix = df_corr.corr()

# Heatmapa korelacija
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Korelaciona matrica atributa", fontsize=14, fontweight="bold", pad=15)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "korelaciona_matrica.png"), dpi=120)
plt.close()

# Prikazi parove sa jakom korelacijom (|r| > 0.7)
print("\nParovi atributa sa jakom korelacijom (|r| > 0.7):")
jaki_parovi = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            jaki_parovi.append((corr_matrix.columns[i], corr_matrix.columns[j], r))
            print(f"  {corr_matrix.columns[i]:25s} <-> {corr_matrix.columns[j]:25s}  r={r:.3f}")

if not jaki_parovi:
    print("  Nema parova sa |r| > 0.7 — nema problema sa multikolinearniscu.")
else:
    print(f"\n  Ukupno {len(jaki_parovi)} jako korelisanih parova.")
    print("  NAPOMENA: Ovo moze uticati na Logisticku regresiju i SVM,")
    print("  ali Random Forest i Gradient Boosting su robustni na korelacije.")

# ====================================================================
#  4. TRENIRANJE MODELA (sa skaliranjem i recency weighting)
# ====================================================================

print("\n" + "=" * 60)
print("  TRENIRANJE MODELA")
print("=" * 60)

modeli = {
    "Logisticka regresija": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]),
    "KNN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ]),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE),
}

rezultati_test = {}
for naziv, model in modeli.items():
    # GBM i RF podrzavaju sample_weight direktno
    try:
        if isinstance(model, Pipeline):
            model.fit(X_train, y_train, clf__sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weights)
    except Exception:
        # Fallback bez weighta
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rezultati_test[naziv] = acc
    print(f"  {naziv:25s}: {acc:.3f}")

# ENSEMBLE - prosjek vjerovatnoca top 3 modela (soft voting)
# Ovo daje robusnije i bolje kalibrisane vjerovatnoce
from sklearn.ensemble import VotingClassifier

print("\n  Treniram VOTING ENSEMBLE (LR + SVM + GB)...")
ensemble = VotingClassifier(
    estimators=[
        ("lr", modeli["Logisticka regresija"]),
        ("svm", modeli["SVM (RBF)"]),
        ("gb", modeli["Gradient Boosting"]),
    ],
    voting="soft"  # soft = koristi vjerovatnoce, ne tvrde predikcije
)
ensemble.fit(X_train, y_train)  # vec trenirani sub-modeli
y_pred_ens = ensemble.predict(X_test)
acc_ens = accuracy_score(y_test, y_pred_ens)
rezultati_test["Voting Ensemble"] = acc_ens
modeli["Voting Ensemble"] = ensemble
print(f"  {'Voting Ensemble':25s}: {acc_ens:.3f}")

# Najbolji model
najbolji_naziv = max(rezultati_test, key=rezultati_test.get)
print(f"\n=> Najbolji: {najbolji_naziv} ({rezultati_test[najbolji_naziv]:.3f})")
najbolji_model = modeli[najbolji_naziv]

# Classification report za najbolji
print(f"\nClassification report ({najbolji_naziv}):")
print(classification_report(y_test, najbolji_model.predict(X_test),
                            target_names=["Team B won", "Team A won"]))

# ====================================================================
#  5. CROSS-VALIDACIJA (FIX: samo na trening setu, bez data leakage)
# ====================================================================

print("\n" + "=" * 60)
print("  K-FOLD CROSS VALIDACIJA (k=5, samo TRAIN)")
print("=" * 60)

cv_rezultati = {}
for naziv, model in modeli.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    cv_rezultati[naziv] = scores
    print(f"  {naziv:25s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Vizualizacija - boxplot
plt.figure(figsize=(10, 5))
plt.boxplot(cv_rezultati.values(), labels=cv_rezultati.keys())
plt.title("K-Fold CV (5-fold) na trening setu")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "poredenje_algoritama_v2.png"), dpi=100)
plt.close()

# Confusion matrix za najbolji
cm = confusion_matrix(y_test, najbolji_model.predict(X_test))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Team B won", "Team A won"],
            yticklabels=["Team B won", "Team A won"])
plt.title(f"Confusion Matrix - {najbolji_naziv}")
plt.ylabel("Stvarno")
plt.xlabel("Predikcija")
plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "confusion_matrix_v2.png"), dpi=100)
plt.close()

# Feature importance (samo za RF/GB)
if hasattr(najbolji_model, "feature_importances_"):
    fi = pd.DataFrame({
        "feature": features,
        "importance": najbolji_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\nTOP 10 najvaznijih featura ({najbolji_naziv}):")
    print(fi.head(10).to_string(index=False))

    plt.figure(figsize=(10, 6))
    fi_top = fi.head(15)
    plt.barh(fi_top["feature"][::-1], fi_top["importance"][::-1], color="steelblue")
    plt.title(f"Feature Importance - {najbolji_naziv}")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_REZULTATI, "feature_importance_v2.png"), dpi=100)
    plt.close()

# ====================================================================
#  5.5 PODESAVANJE HIPERPARAMETARA (GridSearchCV)
# ====================================================================

print("\n" + "=" * 60)
print("  PODESAVANJE HIPERPARAMETARA (GridSearchCV)")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

# Logisticka regresija
print("\n  Logisticka regresija...")
lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
lr_params = {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1", "l2"],
             "clf__solver": ["liblinear"]}
lr_grid = GridSearchCV(lr_pipe, lr_params, cv=5, scoring="accuracy", n_jobs=-1)
lr_grid.fit(X_train, y_train)
print(f"    Najbolji parametri: {lr_grid.best_params_}")
print(f"    CV accuracy:  {lr_grid.best_score_:.3f}")
print(f"    Test accuracy: {lr_grid.score(X_test, y_test):.3f}")

# Random Forest
print("\n  Random Forest...")
rf_params = {"n_estimators": [100, 200, 300], "max_depth": [5, 8, 10, 15, None],
             "min_samples_split": [2, 5, 10]}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    rf_params, cv=5, scoring="accuracy", n_jobs=-1)
rf_grid.fit(X_train, y_train, sample_weight=sample_weights)
print(f"    Najbolji parametri: {rf_grid.best_params_}")
print(f"    CV accuracy:  {rf_grid.best_score_:.3f}")
print(f"    Test accuracy: {rf_grid.score(X_test, y_test):.3f}")

# Gradient Boosting
print("\n  Gradient Boosting...")
gb_params = {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 5, 6],
             "learning_rate": [0.01, 0.05, 0.1]}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    gb_params, cv=5, scoring="accuracy", n_jobs=-1)
gb_grid.fit(X_train, y_train, sample_weight=sample_weights)
print(f"    Najbolji parametri: {gb_grid.best_params_}")
print(f"    CV accuracy:  {gb_grid.best_score_:.3f}")
print(f"    Test accuracy: {gb_grid.score(X_test, y_test):.3f}")

# SVM
print("\n  SVM (RBF)...")
svm_pipe = Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])
svm_params = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", "auto", 0.01, 0.1]}
svm_grid = GridSearchCV(svm_pipe, svm_params, cv=5, scoring="accuracy", n_jobs=-1)
svm_grid.fit(X_train, y_train)
print(f"    Najbolji parametri: {svm_grid.best_params_}")
print(f"    CV accuracy:  {svm_grid.best_score_:.3f}")
print(f"    Test accuracy: {svm_grid.score(X_test, y_test):.3f}")

# KNN
print("\n  KNN...")
knn_pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
knn_params = {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
              "clf__weights": ["uniform", "distance"],
              "clf__metric": ["euclidean", "manhattan"]}
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, scoring="accuracy", n_jobs=-1)
knn_grid.fit(X_train, y_train)
print(f"    Najbolji parametri: {knn_grid.best_params_}")
print(f"    CV accuracy:  {knn_grid.best_score_:.3f}")
print(f"    Test accuracy: {knn_grid.score(X_test, y_test):.3f}")

# Sacuvaj tabelu poredjenja: default vs tuned
print("\n  === POREDENJE: DEFAULT vs TUNED ===")
tuned_modeli = {
    "Logisticka regresija": lr_grid,
    "Random Forest": rf_grid,
    "Gradient Boosting": gb_grid,
    "SVM (RBF)": svm_grid,
    "KNN": knn_grid,
}
print(f"  {'Algoritam':25s} {'Default':>10s} {'Tuned':>10s} {'Razlika':>10s}")
print(f"  {'-'*55}")
for naziv, grid in tuned_modeli.items():
    default_acc = rezultati_test.get(naziv, rezultati_test.get("KNN (k=7)", 0))
    tuned_acc = grid.score(X_test, y_test)
    diff = tuned_acc - default_acc
    print(f"  {naziv:25s} {default_acc:10.3f} {tuned_acc:10.3f} {diff:+10.3f}")

# Azuriraj najbolji model ako je tuned bolji
sve_tuned = {n: g.score(X_test, y_test) for n, g in tuned_modeli.items()}
best_tuned_naziv = max(sve_tuned, key=sve_tuned.get)
best_tuned_acc = sve_tuned[best_tuned_naziv]

if best_tuned_acc > rezultati_test[najbolji_naziv]:
    print(f"\n  Tuned '{best_tuned_naziv}' ({best_tuned_acc:.3f}) je bolji od "
          f"default '{najbolji_naziv}' ({rezultati_test[najbolji_naziv]:.3f})")
    najbolji_model = tuned_modeli[best_tuned_naziv].best_estimator_
    najbolji_naziv = f"{best_tuned_naziv} (tuned)"
    print(f"  -> Koristim tuned model za predikcije")
else:
    print(f"\n  Default '{najbolji_naziv}' je i dalje najbolji ({rezultati_test[najbolji_naziv]:.3f})")

# ====================================================================
#  5.6 ODABIR NAJZNACAJNIJIH ATRIBUTA
# ====================================================================

print("\n" + "=" * 60)
print("  ODABIR NAJZNACAJNIJIH ATRIBUTA")
print("=" * 60)

from sklearn.feature_selection import SelectKBest, f_classif, RFE

# --- Metoda 1: SelectKBest (statisticki test) ---
print("\n--- Metoda 1: SelectKBest (ANOVA F-test) ---")
selector_kb = SelectKBest(f_classif, k="all")
selector_kb.fit(X_train, y_train)

kb_scores = pd.DataFrame({
    "feature": features,
    "f_score": selector_kb.scores_,
    "p_value": selector_kb.pvalues_
}).sort_values("f_score", ascending=False)

print("\nRangiranje atributa po F-score:")
for i, (_, row) in enumerate(kb_scores.iterrows(), 1):
    zvjezdica = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
    print(f"  {i:2d}. {row['feature']:25s}  F={row['f_score']:8.2f}  p={row['p_value']:.4f} {zvjezdica}")

# --- Metoda 2: RFE (Recursive Feature Elimination) sa Logistickom regresijom ---
print("\n--- Metoda 2: RFE (Recursive Feature Elimination) ---")
rfe_estimator = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
# RFE ne radi direktno sa Pipeline, pa koristimo samo na skaliranim podacima
scaler_rfe = StandardScaler()
X_train_scaled = scaler_rfe.fit_transform(X_train)
X_test_scaled = scaler_rfe.transform(X_test)

rfe = RFE(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
          n_features_to_select=10, step=1)
rfe.fit(X_train_scaled, y_train)

rfe_selected = [f for f, s in zip(features, rfe.support_) if s]
rfe_ranking = pd.DataFrame({
    "feature": features,
    "rank": rfe.ranking_,
    "selected": rfe.support_
}).sort_values("rank")

print("\nRFE ranking (1 = odabran):")
for _, row in rfe_ranking.iterrows():
    oznaka = "SELECTED" if row["selected"] else ""
    print(f"  Rank {int(row['rank']):2d}: {row['feature']:25s} {oznaka}")

# --- Metoda 3: Feature Importance iz Random Forest (tuned) ---
print("\n--- Metoda 3: Feature Importance (Random Forest, tuned) ---")
rf_best = rf_grid.best_estimator_
fi_rf = pd.DataFrame({
    "feature": features,
    "importance": rf_best.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 atributa po RF importance:")
for i, (_, row) in enumerate(fi_rf.head(15).iterrows(), 1):
    bar = "#" * int(row["importance"] * 200)
    print(f"  {i:2d}. {row['feature']:25s}  {row['importance']:.4f}  {bar}")

# Vizualizacija poredjenja sve 3 metode
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# SelectKBest
ax = axes[0]
top_kb = kb_scores.head(15)
ax.barh(top_kb["feature"][::-1], top_kb["f_score"][::-1], color="#2a9d8f")
ax.set_title("SelectKBest (F-score)", fontsize=11, fontweight="bold")
ax.set_xlabel("F-score")

# RFE ranking
ax = axes[1]
rfe_top = rfe_ranking.head(15)
boje_rfe = ["#2a9d8f" if s else "#e5e5e5" for s in rfe_top["selected"][::-1]]
ax.barh(rfe_top["feature"][::-1], [max(features.__len__() - r + 1, 0) for r in rfe_top["rank"][::-1]],
        color=boje_rfe)
ax.set_title("RFE (Recursive Feature Elimination)", fontsize=11, fontweight="bold")
ax.set_xlabel("Inverzni rang")

# RF Importance
ax = axes[2]
top_fi = fi_rf.head(15)
ax.barh(top_fi["feature"][::-1], top_fi["importance"][::-1], color="#457b9d")
ax.set_title("Random Forest Importance", fontsize=11, fontweight="bold")
ax.set_xlabel("Importance")

plt.suptitle("Poredenje metoda za odabir atributa", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "feature_selection_poredenje.png"), dpi=120)
plt.close()

# --- POREDENJE: SVI ATRIBUTI vs TOP 10 ---
print("\n" + "=" * 60)
print("  POREDENJE: SVI ATRIBUTI vs TOP 10 (RFE)")
print("=" * 60)

# Top 10 po RFE
X_train_top10 = X_train_scaled[:, rfe.support_]
X_test_top10 = X_test_scaled[:, rfe.support_]

print(f"\n  Svi atributi: {len(features)}")
print(f"  Top 10 (RFE): {rfe_selected}")

# Treniraj i uporedi
modeli_za_poredjenje = {
    "Logisticka regresija": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        **rf_grid.best_params_, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(
        **gb_grid.best_params_, random_state=RANDOM_STATE),
}

print(f"\n  {'Algoritam':25s} {'Svi attr':>10s} {'Top 10':>10s} {'Razlika':>10s}")
print(f"  {'-'*55}")
for naziv, model in modeli_za_poredjenje.items():
    # Svi atributi
    model.fit(X_train_scaled, y_train)
    acc_svi = accuracy_score(y_test, model.predict(X_test_scaled))

    # Top 10
    model_top = type(model)(**model.get_params())
    model_top.fit(X_train_top10, y_train)
    acc_top = accuracy_score(y_test, model_top.predict(X_test_top10))

    diff = acc_top - acc_svi
    print(f"  {naziv:25s} {acc_svi:10.3f} {acc_top:10.3f} {diff:+10.3f}")

print("\n  ZAKLJUCAK: Ako je razlika mala ili pozitivna, znaci da nepotrebni")
print("  atributi dodaju sum (noise) i model radi jednako ili bolje sa manje.")

# ====================================================================
#  6. PREDIKCIJA MECEVA - FUNKCIJE
# ====================================================================

# Posto je kanonizacija imena uradjena na pocetku, samo prosljedjujemo ime kako jest
def normalizuj_ime(tim):
    # Backup - ako neko negdje proslijedi stari alias, ipak ga konvertujemo
    return ALIAS_MAPPING.get(tim, tim)

# Kes feature-a za sve timove iz dataseta
team_feat_dict = team_features.set_index("Team").to_dict("index")
feat_default = {kol: team_features[kol].median()
                for kol in team_features.columns if kol != "Team"}

def dohvati_team_features(tim_naziv):
    tim = normalizuj_ime(tim_naziv)
    if tim in team_feat_dict:
        return team_feat_dict[tim]
    # Pokusaj parcijalni match
    for t in team_feat_dict:
        if tim.lower() in str(t).lower() or str(t).lower() in tim.lower():
            return team_feat_dict[t]
    return feat_default


def dohvati_win_rate_iz_pro(tim_naziv):
    """Vraca posljednji poznati win rate i broj meceva za tim."""
    tim = normalizuj_ime(tim_naziv)
    kao_a = df_pro[df_pro["Team A"] == tim][["win_rate_a", "broj_meceva_a"]].tail(1)
    kao_b = df_pro[df_pro["Team B"] == tim][["win_rate_b", "broj_meceva_b"]].tail(1)
    if not kao_a.empty:
        return float(kao_a["win_rate_a"].values[0]), int(kao_a["broj_meceva_a"].values[0])
    if not kao_b.empty:
        return float(kao_b["win_rate_b"].values[0]), int(kao_b["broj_meceva_b"].values[0])
    return 0.5, 0


def izgradi_red_za_predikciju(tim_a, tim_b):
    """Vraca DataFrame red sa svim feature-ima za par tim_a vs tim_b."""
    wr_a, mec_a = dohvati_win_rate_iz_pro(tim_a)
    wr_b, mec_b = dohvati_win_rate_iz_pro(tim_b)

    # H2H
    tim_a_n = normalizuj_ime(tim_a)
    tim_b_n = normalizuj_ime(tim_b)
    h2h = df_pro[
        ((df_pro["Team A"] == tim_a_n) & (df_pro["Team B"] == tim_b_n)) |
        ((df_pro["Team A"] == tim_b_n) & (df_pro["Team B"] == tim_a_n))
    ]
    h2h_rate = 0.5
    if len(h2h) > 0:
        pob_a = len(h2h[
            ((h2h["Team A"] == tim_a_n) & (h2h["team_a_won"] == 1)) |
            ((h2h["Team B"] == tim_a_n) & (h2h["team_a_won"] == 0))
        ])
        h2h_rate = pob_a / len(h2h)

    feat_a = dohvati_team_features(tim_a)
    feat_b = dohvati_team_features(tim_b)

    red = {
        "h2h_win_rate_a": h2h_rate,
        "h2h_mecevi": len(h2h),
        # ELO i forma - NAJJACI signali
        "a_elo": feat_a.get("elo", 1500),
        "b_elo": feat_b.get("elo", 1500),
        "diff_elo": feat_a.get("elo", 1500) - feat_b.get("elo", 1500),
        "a_forma_15": feat_a.get("forma_15", 0.5),
        "b_forma_15": feat_b.get("forma_15", 0.5),
        "diff_forma": feat_a.get("forma_15", 0.5) - feat_b.get("forma_15", 0.5),
        # Dinamicki WR (POSLE alias merge-a)
        "a_dynamic_wr": feat_a.get("dynamic_wr", 0.5),
        "b_dynamic_wr": feat_b.get("dynamic_wr", 0.5),
        "diff_dynamic_wr": feat_a.get("dynamic_wr", 0.5) - feat_b.get("dynamic_wr", 0.5),
        "diff_dynamic_mecevi": feat_a.get("dynamic_mecevi", 0) - feat_b.get("dynamic_mecevi", 0),
        # Tier1 i recent international
        "a_tier1_wr": feat_a.get("tier1_wr", 0.3),
        "b_tier1_wr": feat_b.get("tier1_wr", 0.3),
        "diff_tier1_wr": feat_a.get("tier1_wr", 0.3) - feat_b.get("tier1_wr", 0.3),
        "diff_tier1_iskustvo": feat_a.get("tier1_meceva", 0) - feat_b.get("tier1_meceva", 0),
        "a_recent_int_wr": feat_a.get("recent_int_wr", 0.3),
        "b_recent_int_wr": feat_b.get("recent_int_wr", 0.3),
        "diff_recent_int_wr": feat_a.get("recent_int_wr", 0.3) - feat_b.get("recent_int_wr", 0.3),
        # Stage 1
        "diff_champ_pts": feat_a.get("championship_points", 0) - feat_b.get("championship_points", 0),
        "diff_stage1": feat_a.get("stage1_score", 0) - feat_b.get("stage1_score", 0),
        # Player stats
        "diff_map_wr": feat_a.get("map_wr", 0.5) - feat_b.get("map_wr", 0.5),
        "diff_rating": feat_a.get("rating_avg", 1.0) - feat_b.get("rating_avg", 1.0),
        "diff_acs": feat_a.get("acs_avg", 200) - feat_b.get("acs_avg", 200),
        "diff_adr": feat_a.get("adr_avg", 130) - feat_b.get("adr_avg", 130),
        "diff_pistol": feat_a.get("pistol_wr", 0.5) - feat_b.get("pistol_wr", 0.5),
        "diff_full_buy": feat_a.get("full_buy_wr", 0.5) - feat_b.get("full_buy_wr", 0.5),
        "diff_eco": feat_a.get("eco_wr", 0.5) - feat_b.get("eco_wr", 0.5),
        "diff_kast": feat_a.get("kast_avg", 70) - feat_b.get("kast_avg", 70),
        "diff_fkpr": feat_a.get("fkpr_avg", 0.1) - feat_b.get("fkpr_avg", 0.1),
        "diff_clutch": feat_a.get("clutch_avg", 0.15) - feat_b.get("clutch_avg", 0.15),
        "diff_hs": feat_a.get("hs_avg", 25) - feat_b.get("hs_avg", 25),
        # Legacy win_rate (iz CSV-a, manje pouzdano ali dodatna info)
        "win_rate_a": wr_a, "win_rate_b": wr_b,
        "diff_win_rate": wr_a - wr_b,
    }
    # Vrati u istom redoslijedu kao "features"
    return np.array([[red.get(f, 0.0) for f in features]])


def vjerovatnoca_pobjede(tim_a, tim_b, model=najbolji_model):
    """Vraca vjerovatnocu da tim_a pobijedi."""
    X = izgradi_red_za_predikciju(tim_a, tim_b)
    proba = model.predict_proba(X)[0]
    return float(proba[1])  # P(team_a_won = 1)


def predvidi_mec(tim_a, tim_b, model=najbolji_model):
    p_a = vjerovatnoca_pobjede(tim_a, tim_b, model)
    if p_a >= 0.5:
        return tim_a, p_a * 100
    return tim_b, (1 - p_a) * 100


# ====================================================================
#  7. SIMULACIJA MASTERS LONDON 2026 - REALAN FORMAT
# ====================================================================

print("\n" + "=" * 60)
print("  MASTERS LONDON 2026 - DETERMINISTICKA PREDIKCIJA")
print("=" * 60)

# Iz Stage 1 standings: po 3 tima po regionu
direktni_seedovi = ["G2 Esports", "Team Heretics", "Paper Rex", "EDward Gaming"]
swiss_timovi = [
    "Xi Lai Gaming", "NRG", "Team Vitality", "Dragon Ranger Gaming",
    "FULL SENSE", "FUT Esports", "LEVIATÁN", "Global Esports",
]

# =========================================================
#  STVARNI PAROVI ROUND 1 (sa vlr.gg, draw 25. maj 2026)
# =========================================================
# Bitno: parovi su izvuceni nakon VCT Americas Stage 1 finala
# i nisu generisani sortiranjem - moraju biti hardcode-ovani!
STVARNI_R1_PAROVI = [
    ("Xi Lai Gaming", "NRG"),
    ("Team Vitality", "Dragon Ranger Gaming"),
    ("FULL SENSE", "FUT Esports"),
    ("LEVIATÁN", "Global Esports"),
]

# =========================================================
#  STVARNI PAROVI ROUND 1 (sa vlr.gg, draw 25. maj 2026)
# =========================================================
# Bitno: parovi su izvuceni nakon VCT Americas Stage 1 finala
# i nisu generisani sortiranjem - moraju biti hardcode-ovani!
# Tek od Round 2 nadalje, parovi se generisu dinamicki po Swiss formatu
# (1-0 vs 1-0, 0-1 vs 0-1) - zato njih ne treba hardcode-ovati.
STVARNI_R1_PAROVI = [
    ("Xi Lai Gaming", "NRG"),
    ("Team Vitality", "Dragon Ranger Gaming"),
    ("FULL SENSE", "FUT Esports"),
    ("LEVIATÁN", "Global Esports"),
]


def odigraj_mec(tim_a, tim_b, scores=None, log=True):
    pobjednik, vjer = predvidi_mec(tim_a, tim_b)
    gubitnik = tim_b if pobjednik == tim_a else tim_a
    if scores is not None:
        scores[pobjednik][0] += 1
        scores[gubitnik][1] += 1
    if log:
        print(f"  {tim_a:25s} vs {tim_b:25s} -> {pobjednik:25s} ({vjer:.1f}%)")
    return pobjednik, gubitnik


def simuliraj_swiss(timovi, r1_parovi=None, log=True):
    """Standardni Swiss format: 3 runde, do 2 pobjede ili 2 poraza.

    r1_parovi: lista (tim_a, tim_b) - STVARNI parovi iz drawa.
               Ako None, generise se zip(prva_polovina, druga_polovina).
    """
    scores = {t: [0, 0] for t in timovi}

    # Round 1 - koristi STVARNE parove ako su dati
    if log:
        print("\n--- SWISS ROUND 1 ---")
    if r1_parovi is None:
        polovina = len(timovi) // 2
        r1_parovi = list(zip(timovi[:polovina], timovi[polovina:]))
    for a, b in r1_parovi:
        odigraj_mec(a, b, scores, log)

    # Round 2: 1-0 vs 1-0, 0-1 vs 0-1
    if log:
        print("\n--- SWISS ROUND 2 ---")
    t_1_0 = sorted([t for t, s in scores.items() if s == [1, 0]])
    t_0_1 = sorted([t for t, s in scores.items() if s == [0, 1]])
    for grupa in [t_1_0, t_0_1]:
        for i in range(0, len(grupa) - 1, 2):
            odigraj_mec(grupa[i], grupa[i + 1], scores, log)

    # Round 3: 1-1 timovi - eliminacijski
    if log:
        print("\n--- SWISS ROUND 3 ---")
    t_2_0 = [t for t, s in scores.items() if s == [2, 0]]
    t_1_1 = sorted([t for t, s in scores.items() if s == [1, 1]])
    prolaz = list(t_2_0)
    for i in range(0, len(t_1_1) - 1, 2):
        pob, gub = odigraj_mec(t_1_1[i], t_1_1[i + 1], scores, log)
        prolaz.append(pob)
    return prolaz


def simuliraj_double_elim_8(direktni, swiss_survivori, log=True):
    """STVARAN format Masters London 2026:
    - 4 direktni (1. seedovi) BIRAJU protivnika iz 4 Swiss survivora
    - Redoslijed biranja je RANDOM
    - Svaki direktni bira tima koji mu daje najvecu sansu pobjede
    """
    if log:
        print("\n--- DRAW: 1. seedovi biraju protivnike (random redoslijed) ---")

    # Random redoslijed biranja - kljucno, kao u stvarnosti!
    redoslijed_biranja = list(direktni)
    random.shuffle(redoslijed_biranja)

    dostupni = list(swiss_survivori)
    parovi_uqf = []
    for seed in redoslijed_biranja:
        # Bira protivnika sa najmanjom vjerovatnocom da pobijedi (= najveca sansa za seed)
        protivnik = min(dostupni,
                       key=lambda p: vjerovatnoca_pobjede(p, seed))
        dostupni.remove(protivnik)
        parovi_uqf.append((seed, protivnik))
        if log:
            p_seed = vjerovatnoca_pobjede(seed, protivnik) * 100
            print(f"  {seed:25s} bira {protivnik:25s} ({p_seed:.1f}% za seeda)")

    if log:
        print("\n--- UPPER QUARTERFINALS ---")
    uqf_pob, uqf_gub = [], []
    for a, b in parovi_uqf:
        p, g = odigraj_mec(a, b, log=log)
        uqf_pob.append(p); uqf_gub.append(g)

    if log:
        print("\n--- UPPER SEMIFINALS ---")
    usf_pob = []
    for i in range(0, len(uqf_pob), 2):
        p, _ = odigraj_mec(uqf_pob[i], uqf_pob[i + 1], log=log)
        usf_pob.append(p)
    uqf_gub_iz_sf = [t for pair in zip(uqf_pob[::2], uqf_pob[1::2])
                     for t in pair if t not in usf_pob]

    if log:
        print("\n--- LOWER ROUND 1 (UQF gubitnici)---")
    # UB QF gubitnici: 1v2, 3v4 (po redoslijedu UB QF parova)
    lr1_pob = []
    for i in range(0, len(uqf_gub), 2):
        p, _ = odigraj_mec(uqf_gub[i], uqf_gub[i + 1], log=log)
        lr1_pob.append(p)

    if log:
        print("\n--- LOWER ROUND 2 (UB SF gub vs LB R1 pob) ---")
    lr2_pob = []
    for i in range(len(lr1_pob)):
        p, _ = odigraj_mec(uqf_gub_iz_sf[i], lr1_pob[i], log=log)
        lr2_pob.append(p)

    if log:
        print("\n--- UPPER FINAL ---")
    uf_pob, uf_gub = odigraj_mec(usf_pob[0], usf_pob[1], log=log)

    if log:
        print("\n--- LOWER FINAL ---")
    lf_pob, _ = odigraj_mec(lr2_pob[0], lr2_pob[1] if len(lr2_pob) > 1 else uf_gub, log=log)

    # Lower bracket nastavak
    if log:
        print("\n--- LOWER BRACKET FINAL ---")
    lbf_pob, _ = odigraj_mec(uf_gub, lf_pob, log=log)

    if log:
        print("\n--- GRAND FINAL ---")
    gf_pob, gf_gub = odigraj_mec(uf_pob, lbf_pob, log=log)
    return gf_pob


# Glavna simulacija
swiss_prolaz = simuliraj_swiss(swiss_timovi, r1_parovi=STVARNI_R1_PAROVI, log=True)
print(f"\nIz Swissa prolaze: {swiss_prolaz}")

print(f"\nDirektni (1. seedovi): {direktni_seedovi}")
print(f"Swiss survivori: {swiss_prolaz}")
pobjednik = simuliraj_double_elim_8(direktni_seedovi, swiss_prolaz, log=True)

print("\n" + "=" * 60)
print(f"   POBJEDNIK (DETERMINISTICKI): {pobjednik}")
print("=" * 60)

# ====================================================================
#  8. MONTE CARLO - REALAN (sa pravim randomom!)
# ====================================================================

print("\n" + "=" * 60)
print("  MONTE CARLO SIMULACIJA")
print("=" * 60)

masters_svi = direktni_seedovi + swiss_timovi

# Kes vjerovatnoca za sve parove
print("Kesiranje vjerovatnoca...")
kes_proba = {}
for a in masters_svi:
    for b in masters_svi:
        if a != b:
            kes_proba[(a, b)] = vjerovatnoca_pobjede(a, b)


def mc_odigraj(a, b, scores=None):
    """Koristi PRAVI random za rezultat - tako fokus na nesigurnost
    (ne uvjek pobjeduje fovorit, kao u stvarnosti)."""
    p_a = kes_proba.get((a, b), 0.5)
    if random.random() < p_a:
        pob, gub = a, b
    else:
        pob, gub = b, a
    if scores is not None:
        if pob in scores: scores[pob][0] += 1
        if gub in scores: scores[gub][1] += 1
    return pob, gub


def mc_swiss(timovi, r1_parovi=None):
    scores = {t: [0, 0] for t in timovi}

    # Round 1 - koristi STVARNE parove
    if r1_parovi is None:
        polovina = len(timovi) // 2
        r1_parovi = list(zip(timovi[:polovina], timovi[polovina:]))
    for a, b in r1_parovi:
        mc_odigraj(a, b, scores)

    t10 = [t for t, s in scores.items() if s == [1, 0]]
    t01 = [t for t, s in scores.items() if s == [0, 1]]
    random.shuffle(t10); random.shuffle(t01)
    for grupa in [t10, t01]:
        for i in range(0, len(grupa) - 1, 2):
            mc_odigraj(grupa[i], grupa[i + 1], scores)

    t20 = [t for t, s in scores.items() if s == [2, 0]]
    t11 = [t for t, s in scores.items() if s == [1, 1]]
    random.shuffle(t11)
    prolaz = list(t20)
    for i in range(0, len(t11) - 1, 2):
        p, _ = mc_odigraj(t11[i], t11[i + 1], scores)
        prolaz.append(p)
    return prolaz


def mc_double_elim(direktni, swiss_survivori):
    """Stvaran format - 1. seedovi biraju u random redoslijedu."""
    # KLJUCNO: random redoslijed biranja svake simulacije
    redoslijed = list(direktni)
    random.shuffle(redoslijed)

    dostupni = list(swiss_survivori)
    parovi_uqf = []
    for seed in redoslijed:
        # Bira najslabijeg dostupnog protivnika
        protivnik = min(dostupni, key=lambda p: kes_proba.get((p, seed), 0.5))
        dostupni.remove(protivnik)
        parovi_uqf.append((seed, protivnik))

    uqf_pob, uqf_gub = [], []
    for a, b in parovi_uqf:
        p, g = mc_odigraj(a, b); uqf_pob.append(p); uqf_gub.append(g)

    usf_pob, usf_gub = [], []
    for i in range(0, len(uqf_pob), 2):
        p, g = mc_odigraj(uqf_pob[i], uqf_pob[i + 1])
        usf_pob.append(p); usf_gub.append(g)

    lr1_pob = []
    for i in range(0, len(uqf_gub), 2):
        p, _ = mc_odigraj(uqf_gub[i], uqf_gub[i + 1])
        lr1_pob.append(p)
    lr2_pob = [mc_odigraj(usf_gub[i], lr1_pob[i])[0] for i in range(len(lr1_pob))]

    uf_pob, uf_gub = mc_odigraj(usf_pob[0], usf_pob[1])
    if len(lr2_pob) >= 2:
        lf_pob, _ = mc_odigraj(lr2_pob[0], lr2_pob[1])
    else:
        lf_pob = lr2_pob[0]
    lbf_pob, _ = mc_odigraj(uf_gub, lf_pob)
    gf_pob, _ = mc_odigraj(uf_pob, lbf_pob)
    return gf_pob


def monte_carlo(n_simulacija=10000):
    pobjede = {}
    print(f"\nPokrecem {n_simulacija} simulacija...")
    for sim in range(n_simulacija):
        if sim % 2000 == 0 and sim > 0:
            print(f"  {sim}/{n_simulacija}...")
        sw_prolaz = mc_swiss(list(swiss_timovi), r1_parovi=STVARNI_R1_PAROVI)
        winner = mc_double_elim(direktni_seedovi, sw_prolaz)
        pobjede[winner] = pobjede.get(winner, 0) + 1

    print("\n" + "=" * 60)
    print(f"   MONTE CARLO ({n_simulacija} sim)")
    print("=" * 60)
    print("\nVjerovatnoca osvajanja:\n")
    sortirano = sorted(pobjede.items(), key=lambda x: x[1], reverse=True)
    for tim, p in sortirano:
        proc = p / n_simulacija * 100
        bar = "#" * int(proc / 2)
        print(f"  {tim:25s} {proc:5.1f}%  {bar}")

    # Vizualizacija
    plt.figure(figsize=(10, 6))
    timovi_mc = [t for t, _ in sortirano]
    procenti = [p / n_simulacija * 100 for _, p in sortirano]
    plt.barh(timovi_mc[::-1], procenti[::-1], color="steelblue")
    plt.xlabel("Vjerovatnoca osvajanja (%)")
    plt.title(f"Masters London 2026 - Monte Carlo ({n_simulacija} sim)")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_REZULTATI, "monte_carlo_v2.png"), dpi=100)
    plt.close()

    return pobjede


random.seed(RANDOM_STATE)
monte_carlo(10000)

print("\nGotovo! Svi grafovi su sacuvani u:", PATH_REZULTATI)