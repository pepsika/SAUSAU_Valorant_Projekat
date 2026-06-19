"""
====================================================================
   VALORANT VCT - PREDIKCIJA MASTERS LONDON 2026
====================================================================
Glavna skripta - pokrece cijeli pipeline:
  1. Ucitavanje podataka
  2. Feature engineering po timu (ELO, recency, eco, tier-1 itd.)
  3. Priprema dataseta (diff-feature, train/test split)
  4. EDA - korelaciona analiza + detekcija anomalija
  5. Treniranje modela (5 algoritama + voting ensemble)
  6. K-fold cross validacija
  7. Podesavanje hiperparametara (GridSearchCV)
  8. Odabir najznacajnijih atributa (SelectKBest, RFE, RF importance)
  9. Export modela (MatchPredictor) za deployment - app.py
 10. Deterministicka i Monte Carlo simulacija Masters London 2026

Logika je rasporedjena u zasebne module (data_loading, features,
dataset_prep, eda, models, feature_selection, predict, simulate) - ovaj
fajl ih samo poziva redom.
"""

import warnings

import joblib

from config import (
    PATH_MODEL, PATH_REZULTATI, RANDOM_STATE,
    DIREKTNI_SEEDOVI, SWISS_TIMOVI, STVARNI_R1_PAROVI, ALIAS_MAPPING,
)
from data_loading import load_all_data
from features import build_team_features
from dataset_prep import prepare_dataset
from eda import correlation_analysis, outlier_analysis
from models import (
    train_models, cross_validate_models, plot_confusion_matrix,
    plot_feature_importance, tune_hyperparameters,
)
from feature_selection import run_feature_selection
from predict import MatchPredictor
from simulate import simuliraj_masters_london, monte_carlo
import random

warnings.filterwarnings("ignore")


def main():
    # 1. Podaci
    data = load_all_data()

    # 2. Feature engineering po timu
    team_features = build_team_features(data)

    # 3. Priprema dataseta
    ds = prepare_dataset(data["df_pro"], team_features)

    # 4. EDA
    correlation_analysis(ds["X_train"], ds["features"])
    outlier_analysis(ds["X_train"], ds["features"])

    # 5. Treniranje modela
    rez = train_models(ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"], ds["sample_weights"])

    # 6. Cross validacija + vizualizacije
    cross_validate_models(rez["modeli"], ds["X_train"], ds["y_train"])
    plot_confusion_matrix(rez["najbolji_model"], rez["najbolji_naziv"], ds["X_test"], ds["y_test"])
    plot_feature_importance(rez["najbolji_model"], rez["najbolji_naziv"], ds["features"])

    # 7. Hiperparametri
    tuning = tune_hyperparameters(
        ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"], ds["sample_weights"],
        rez["rezultati_test"], rez["najbolji_model"], rez["najbolji_naziv"])
    najbolji_model = tuning["najbolji_model"]
    najbolji_naziv = tuning["najbolji_naziv"]

    # 8. Odabir najznacajnijih atributa
    run_feature_selection(
        ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"], ds["features"],
        tuning["rf_grid"], tuning["gb_grid"])

    # 9. Export modela za deployment (UI/API u app.py)
    print("\n" + "=" * 60)
    print("  EXPORT MODELA")
    print("=" * 60)
    predictor = MatchPredictor(
        model=najbolji_model,
        model_name=najbolji_naziv,
        features=ds["features"],
        team_features=team_features,
        df_pro=data["df_pro"],
        alias_mapping=ALIAS_MAPPING,
    )
    joblib.dump(predictor, PATH_MODEL)
    print(f"  Model sacuvan: {PATH_MODEL}  ({najbolji_naziv})")

    # 10. Simulacija Masters London 2026
    random.seed(RANDOM_STATE)
    simuliraj_masters_london(predictor, DIREKTNI_SEEDOVI, SWISS_TIMOVI, STVARNI_R1_PAROVI, log=True)
    monte_carlo(predictor, DIREKTNI_SEEDOVI, SWISS_TIMOVI, STVARNI_R1_PAROVI, n_simulacija=10000)

    print("\nGotovo! Svi grafovi su sacuvani u:", PATH_REZULTATI)


if __name__ == "__main__":
    main()
