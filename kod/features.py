"""Feature engineering po timu - ekonomija, igracke statistike, mape,
Stage 1 2026, Tier-1 uspjeh, dinamicki win rate i ELO rejting."""

import numpy as np
import pandas as pd

from config import TIER1_TURNIRI, RECENT_INT_TURNIRI
from data_loading import cisto_procenti


def _eco_kategorija_wr(df_eco, tip, naziv):
    sub = df_eco[df_eco["Type"] == tip].copy()
    grupa = sub.groupby("Team").agg(
        won=("Won", "sum"),
        init=("Initiated", "sum")
    ).reset_index()
    grupa[naziv] = grupa["won"] / grupa["init"].replace(0, np.nan)
    return grupa[["Team", naziv]]


def eco_features(df_eco):
    """Pistol/full-buy/eco win rate po timu (FIX: prave kategorije, ne sve 'Won')."""
    df_eco = df_eco.copy()
    df_eco["Initiated"] = pd.to_numeric(df_eco["Initiated"], errors="coerce")
    df_eco["Won"] = pd.to_numeric(df_eco["Won"], errors="coerce")

    # Pistol Won - posebno tretiramo (Initiated nije popunjen, ali je uvjek 2 po mapi)
    pistol_df = df_eco[df_eco["Type"] == "Pistol Won"].groupby("Team").agg(
        pistol_pobjede=("Won", "sum"),
        pistol_mecevi=("Won", "count")
    ).reset_index()
    pistol_df["pistol_wr"] = pistol_df["pistol_pobjede"] / (pistol_df["pistol_mecevi"] * 2)

    full_buy_df = _eco_kategorija_wr(df_eco, "$$$ (won)", "full_buy_wr")
    eco_won_df = _eco_kategorija_wr(df_eco, "Eco (won)", "eco_wr")
    semi_buy_df = _eco_kategorija_wr(df_eco, "$$ (won)", "semi_buy_wr")

    return pistol_df[["Team", "pistol_wr"]], full_buy_df, eco_won_df, semi_buy_df


def player_overview_features(df_overview):
    """Rating/ACS/ADR prosjek po timu (cijeli mec, Side == 'both')."""
    df_overview = df_overview.copy()
    df_overview["Rating"] = pd.to_numeric(df_overview["Rating"], errors="coerce")
    df_overview["Average Combat Score"] = pd.to_numeric(
        df_overview["Average Combat Score"], errors="coerce")
    df_overview["Average Damage Per Round"] = pd.to_numeric(
        df_overview["Average Damage Per Round"], errors="coerce")

    df_ov_both = df_overview[df_overview["Side"] == "both"]
    return df_ov_both.groupby("Team").agg(
        rating_avg=("Rating", "mean"),
        acs_avg=("Average Combat Score", "mean"),
        adr_avg=("Average Damage Per Round", "mean")
    ).reset_index()


def _parsiraj_clutch(s):
    try:
        w, p = str(s).split("/")
        w, p = int(w), int(p)
        return w / p if p > 0 else np.nan
    except Exception:
        return np.nan


def player_advanced_features(df_players):
    """HS%, KAST%, FKPR, clutch rate prosjek po timu."""
    df_players = df_players.copy()
    df_players["Rating"] = pd.to_numeric(df_players["Rating"], errors="coerce")
    df_players["Headshot %"] = cisto_procenti(df_players["Headshot %"]) * 100
    df_players["Kill, Assist, Trade, Survive %"] = cisto_procenti(
        df_players["Kill, Assist, Trade, Survive %"]) * 100
    df_players["First Kills Per Round"] = pd.to_numeric(
        df_players["First Kills Per Round"], errors="coerce")
    df_players["clutch_rate"] = df_players["Clutches (won/played)"].apply(_parsiraj_clutch)

    return df_players.groupby("Teams").agg(
        hs_avg=("Headshot %", "mean"),
        kast_avg=("Kill, Assist, Trade, Survive %", "mean"),
        fkpr_avg=("First Kills Per Round", "mean"),
        clutch_avg=("clutch_rate", "mean")
    ).reset_index().rename(columns={"Teams": "Team"})


def map_win_rate_features(df_maps_scores):
    """Win rate po mapi (agregat preko svih mapa odigranih)."""
    df_maps_scores = df_maps_scores.copy()
    df_maps_scores["team_a_won_map"] = (
        df_maps_scores["Team A Score"] > df_maps_scores["Team B Score"]).astype(int)
    df_maps_scores["team_b_won_map"] = 1 - df_maps_scores["team_a_won_map"]

    wr_a = df_maps_scores.groupby("Team A").agg(
        map_pob_a=("team_a_won_map", "sum"),
        map_uk_a=("team_a_won_map", "count")
    ).reset_index().rename(columns={"Team A": "Team"})

    wr_b = df_maps_scores.groupby("Team B").agg(
        map_pob_b=("team_b_won_map", "sum"),
        map_uk_b=("team_b_won_map", "count")
    ).reset_index().rename(columns={"Team B": "Team"})

    team_map_wr = wr_a.merge(wr_b, on="Team", how="outer").fillna(0)
    team_map_wr["ukupne_pobjede_mapa"] = team_map_wr["map_pob_a"] + team_map_wr["map_pob_b"]
    team_map_wr["ukupno_mapa"] = team_map_wr["map_uk_a"] + team_map_wr["map_uk_b"]
    team_map_wr["map_wr"] = team_map_wr["ukupne_pobjede_mapa"] / team_map_wr["ukupno_mapa"].replace(0, np.nan)
    return team_map_wr[["Team", "map_wr", "ukupno_mapa"]]


def stage1_features(df_stage1):
    """Championship points i Stage 1 2026 placement (kvalifikacije za London)."""
    feat = df_stage1[["team", "championship_points", "placement", "region"]].copy()
    feat.columns = ["Team", "championship_points", "stage1_placement", "region"]
    # Skaliraj placement: 1. mjesto = 1.0, 12. mjesto = 0.0
    feat["stage1_score"] = (13 - feat["stage1_placement"]) / 12.0
    return feat


def _win_rate_a_b(df, win_col_a, win_col_b_name):
    """Pomocna funkcija - win rate kao Team A i kao Team B, spojeno po timu."""
    a_stats = df.groupby("Team A").agg(
        pob_a=(win_col_a, "sum"), mec_a=(win_col_a, "count")
    ).reset_index().rename(columns={"Team A": "Team"})

    df_temp = df.copy()
    df_temp[win_col_b_name] = 1 - df_temp[win_col_a]
    b_stats = df_temp.groupby("Team B").agg(
        pob_b=(win_col_b_name, "sum"), mec_b=(win_col_b_name, "count")
    ).reset_index().rename(columns={"Team B": "Team"})

    return a_stats.merge(b_stats, on="Team", how="outer").fillna(0)


def tier1_features(df_pro):
    """Win rate na Tier-1 turnirima (Champions/Masters) - dokaz istorijskog uspjeha."""
    df_tier1 = df_pro[df_pro["Tournament"].isin(TIER1_TURNIRI)].copy()
    comb = _win_rate_a_b(df_tier1, "team_a_won", "team_b_won")
    comb["tier1_pobjede"] = comb["pob_a"] + comb["pob_b"]
    comb["tier1_meceva"] = comb["mec_a"] + comb["mec_b"]
    comb["tier1_wr"] = comb["tier1_pobjede"] / comb["tier1_meceva"].replace(0, np.nan)
    feat = comb[["Team", "tier1_wr", "tier1_meceva"]]
    # Timovi koji nikad nisu igrali na Tier-1 = signal slabosti, ne 0.5!
    # Stavljamo 0.3 - znaci 'rijetko/nikad na top sceni' = ispod prosjeka
    feat = feat.fillna({"tier1_wr": 0.3, "tier1_meceva": 0})
    print(f"  Tier-1 turniri ucitani: {len(df_tier1)} meceva, "
          f"{len(feat[feat['tier1_meceva'] > 0])} timova sa iskustvom")
    return feat


def recent_international_features(df_pro):
    """Win rate na 2025/2026 Mastersima i Championsima - trenutna forma na top sceni."""
    df_recent = df_pro[df_pro["Tournament"].isin(RECENT_INT_TURNIRI)].copy()
    comb = _win_rate_a_b(df_recent, "team_a_won", "team_b_won")
    comb["recent_int_pobjede"] = comb["pob_a"] + comb["pob_b"]
    comb["recent_int_meceva"] = comb["mec_a"] + comb["mec_b"]
    comb["recent_int_wr"] = comb["recent_int_pobjede"] / comb["recent_int_meceva"].replace(0, np.nan)
    feat = comb[["Team", "recent_int_wr", "recent_int_meceva"]]
    return feat.fillna({"recent_int_wr": 0.3, "recent_int_meceva": 0})


def dynamic_win_rate(df_pro):
    """Win rate POSLE alias-kanonizacije (CSV vrijednosti su racunate PRIJE merga).

    KLJUCNO: win_rate_a/b iz originalnog CSV-a su pogresne za NRG (samo 9
    meceva umjesto 116, jer su "NRG Esports" i "Mega Minors" tretirani kao
    odvojeni timovi prije kanonizacije). Moramo preracunati.
    """
    comb = _win_rate_a_b(df_pro, "team_a_won", "team_b_won_t")
    comb["dynamic_wr"] = (comb["pob_a"] + comb["pob_b"]) / (comb["mec_a"] + comb["mec_b"])
    comb["dynamic_mecevi"] = comb["mec_a"] + comb["mec_b"]
    return comb[["Team", "dynamic_wr", "dynamic_mecevi"]]


def racunaj_elo(df, k_base=32, start_elo=1500):
    """ELO rejting sistem - mjeri RELATIVNU jacinu tima.

    Pobjeda nad jakim timom daje vise nego pobjeda nad slabim, sto prirodno
    razlikuje npr. Champions pobjednika (pobijedio elitne timove) od tima koji
    je skupio pobjede u slabijoj regiji.
    """
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

    forma_dict = {}
    for tim, results in forma.items():
        forma_dict[tim] = sum(results) / len(results) if results else 0.5

    return elo, forma_dict


def build_team_features(data):
    """Spaja sve feature-e po timu u jedan DataFrame (Team + sve kolone).

    `data` je dict iz data_loading.load_all_data().
    """
    print("\n" + "=" * 60)
    print("  KREIRANJE FEATURA PO TIMU")
    print("=" * 60)

    pistol_df, full_buy_df, eco_won_df, semi_buy_df = eco_features(data["df_eco"])
    team_player_stats = player_overview_features(data["df_overview"])
    team_player_advanced = player_advanced_features(data["df_players"])
    team_map_wr = map_win_rate_features(data["df_maps_scores"])
    stage1_feat = stage1_features(data["df_stage1"])
    tier1_feat = tier1_features(data["df_pro"])
    recent_feat = recent_international_features(data["df_pro"])
    dynamic_wr_df = dynamic_win_rate(data["df_pro"])

    print("\n  Dinamicki WR (nakon kanonizacije):")
    for tim in ["NRG", "Paper Rex", "G2 Esports", "Team Vitality", "Team Heretics"]:
        row = dynamic_wr_df[dynamic_wr_df["Team"] == tim]
        if not row.empty:
            print(f"    {tim:25s} WR={row['dynamic_wr'].values[0]:.3f}  "
                  f"meceva={int(row['dynamic_mecevi'].values[0])}")

    elo_dict, forma_dict = racunaj_elo(data["df_pro"])
    elo_df = pd.DataFrame([
        {"Team": t, "elo": e, "forma_15": forma_dict.get(t, 0.5)}
        for t, e in elo_dict.items()
    ])

    print("\n  ELO rejting (top Masters London timovi):")
    london_timovi = ["NRG", "Paper Rex", "G2 Esports", "Team Heretics",
                      "Team Vitality", "EDward Gaming", "LEVIATÁN",
                      "Xi Lai Gaming", "FULL SENSE", "FUT Esports",
                      "Global Esports", "Dragon Ranger Gaming"]
    elo_london = elo_df[elo_df["Team"].isin(london_timovi)].sort_values("elo", ascending=False)
    for _, row in elo_london.iterrows():
        print(f"    {row['Team']:25s} ELO={row['elo']:.0f}  "
              f"forma(15)={row['forma_15']:.3f}")

    team_features = team_map_wr.copy()
    for df_dod in [team_player_stats, team_player_advanced, pistol_df,
                   full_buy_df, eco_won_df, semi_buy_df,
                   stage1_feat[["Team", "championship_points", "stage1_score"]],
                   tier1_feat, recent_feat, dynamic_wr_df, elo_df]:
        team_features = team_features.merge(df_dod, on="Team", how="outer")

    # Popunjavanje NaN sa MEDIANOM (ne 0.5 - 0.5 ima smisla samo za WR, ne za rating!)
    for kol in team_features.columns:
        if kol == "Team":
            continue
        if team_features[kol].dtype in [np.float64, np.int64]:
            team_features[kol] = team_features[kol].fillna(team_features[kol].median())

    print(f"Ukupno timova sa featurima: {len(team_features)}")
    print(f"Broj featura po timu: {team_features.shape[1] - 1}")

    return team_features
