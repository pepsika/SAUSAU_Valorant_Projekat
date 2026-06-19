"""Spajanje team-level featura sa mecevima, izvodjenje diff-featura i
train/test split (po godinama, sa recency weighting)."""

import numpy as np

from config import RECENCY_WEIGHTS

TARGET = "team_a_won"

# (a_kolona, b_kolona, naziv_diff_kolone)
DIFF_FEATURE_PAIRS = [
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

# Konacna lista featura - kombinacija individualnih i razlika
FEATURES = [
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

TRAIN_GODINE = ["vct_2021", "vct_2022", "vct_2023", "vct_2024"]
TEST_GODINE = ["vct_2025", "vct_2026"]


def prepare_dataset(df_pro, team_features):
    """Spaja mecevе sa team-level featurima, izvodi diff-feature i pravi train/test split."""
    print("\n" + "=" * 60)
    print("  PRIPREMA TRENING/TEST")
    print("=" * 60)

    df_full = df_pro.merge(
        team_features.add_prefix("a_").rename(columns={"a_Team": "Team A"}),
        on="Team A", how="left"
    )
    df_full = df_full.merge(
        team_features.add_prefix("b_").rename(columns={"b_Team": "Team B"}),
        on="Team B", how="left"
    )

    for kol in df_full.select_dtypes(include=[np.number]).columns:
        df_full[kol] = df_full[kol].fillna(df_full[kol].median())

    for a_col, b_col, diff_col in DIFF_FEATURE_PAIRS:
        if a_col in df_full.columns and b_col in df_full.columns:
            df_full[diff_col] = df_full[a_col] - df_full[b_col]

    features = [f for f in FEATURES if f in df_full.columns]
    print(f"Ukupno featura u modelu: {len(features)}")

    train = df_full[df_full["godina"].isin(TRAIN_GODINE)].copy()
    test = df_full[df_full["godina"].isin(TEST_GODINE)].copy()

    train["recency_weight"] = train["godina"].map(RECENCY_WEIGHTS)
    sample_weights = train["recency_weight"].values

    X_train = train[features].values
    y_train = train[TARGET].values
    X_test = test[features].values
    y_test = test[TARGET].values

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    return {
        "df_full": df_full,
        "features": features,
        "train": train,
        "test": test,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "sample_weights": sample_weights,
    }
