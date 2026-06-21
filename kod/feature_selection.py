"""Odabir najznacajnijih atributa: SelectKBest, RFE i Random Forest importance,
te poredjenje performansi sa svim atributima naspram top 10."""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # neinteraktivni backend - skripta samo cuva fajlove (savefig), bez prozora
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import PATH_REZULTATI, RANDOM_STATE


def select_k_best(X_train, y_train, features):
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

    return kb_scores


def recursive_feature_elimination(X_train, y_train, X_test, y_test, features, n_select=10):
    print("\n--- Metoda 2: RFE (Recursive Feature Elimination) ---")
    scaler_rfe = StandardScaler()
    X_train_scaled = scaler_rfe.fit_transform(X_train)
    X_test_scaled = scaler_rfe.transform(X_test)

    rfe = RFE(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
              n_features_to_select=n_select, step=1)
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

    return rfe, rfe_ranking, rfe_selected, X_train_scaled, X_test_scaled


def rf_importance(rf_best, features):
    print("\n--- Metoda 3: Feature Importance (Random Forest, tuned) ---")
    fi_rf = pd.DataFrame({
        "feature": features,
        "importance": rf_best.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 15 atributa po RF importance:")
    for i, (_, row) in enumerate(fi_rf.head(15).iterrows(), 1):
        bar = "#" * int(row["importance"] * 200)
        print(f"  {i:2d}. {row['feature']:25s}  {row['importance']:.4f}  {bar}")

    return fi_rf


def plot_feature_selection_comparison(kb_scores, rfe_ranking, fi_rf, n_features):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    ax = axes[0]
    top_kb = kb_scores.head(15)
    ax.barh(top_kb["feature"][::-1], top_kb["f_score"][::-1], color="#2a9d8f")
    ax.set_title("SelectKBest (F-score)", fontsize=11, fontweight="bold")
    ax.set_xlabel("F-score")

    ax = axes[1]
    rfe_top = rfe_ranking.head(15)
    boje_rfe = ["#2a9d8f" if s else "#e5e5e5" for s in rfe_top["selected"][::-1]]
    ax.barh(rfe_top["feature"][::-1],
            [max(n_features - r + 1, 0) for r in rfe_top["rank"][::-1]],
            color=boje_rfe)
    ax.set_title("RFE (Recursive Feature Elimination)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Inverzni rang")

    ax = axes[2]
    top_fi = fi_rf.head(15)
    ax.barh(top_fi["feature"][::-1], top_fi["importance"][::-1], color="#457b9d")
    ax.set_title("Random Forest Importance", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance")

    plt.suptitle("Poredenje metoda za odabir atributa", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_REZULTATI, "feature_selection_poredenje.png"), dpi=120)
    plt.close()


def compare_all_vs_top10(X_train_scaled, X_test_scaled, y_train, y_test, rfe, rfe_selected,
                          features, rf_grid, gb_grid):
    print("\n" + "=" * 60)
    print("  POREDENJE: SVI ATRIBUTI vs TOP 10 (RFE)")
    print("=" * 60)

    X_train_top10 = X_train_scaled[:, rfe.support_]
    X_test_top10 = X_test_scaled[:, rfe.support_]

    print(f"\n  Svi atributi: {len(features)}")
    print(f"  Top 10 (RFE): {rfe_selected}")

    modeli_za_poredjenje = {
        "Logisticka regresija": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            **rf_grid.best_params_, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            **gb_grid.best_params_, random_state=RANDOM_STATE),
    }

    print(f"\n  {'Algoritam':25s} {'Svi attr':>10s} {'Top 10':>10s} {'Razlika':>10s}")
    print(f"  {'-' * 55}")
    poredjenje = {}
    for naziv, model in modeli_za_poredjenje.items():
        model.fit(X_train_scaled, y_train)
        acc_svi = accuracy_score(y_test, model.predict(X_test_scaled))

        model_top = type(model)(**model.get_params())
        model_top.fit(X_train_top10, y_train)
        acc_top = accuracy_score(y_test, model_top.predict(X_test_top10))

        diff = acc_top - acc_svi
        poredjenje[naziv] = {"svi": acc_svi, "top10": acc_top, "razlika": diff}
        print(f"  {naziv:25s} {acc_svi:10.3f} {acc_top:10.3f} {diff:+10.3f}")

    print("\n  ZAKLJUCAK: Ako je razlika mala ili pozitivna, znaci da nepotrebni")
    print("  atributi dodaju sum (noise) i model radi jednako ili bolje sa manje.")

    return poredjenje


def run_feature_selection(X_train, y_train, X_test, y_test, features, rf_grid, gb_grid):
    kb_scores = select_k_best(X_train, y_train, features)
    rfe, rfe_ranking, rfe_selected, X_train_scaled, X_test_scaled = recursive_feature_elimination(
        X_train, y_train, X_test, y_test, features)
    fi_rf = rf_importance(rf_grid.best_estimator_, features)
    plot_feature_selection_comparison(kb_scores, rfe_ranking, fi_rf, len(features))
    poredjenje = compare_all_vs_top10(
        X_train_scaled, X_test_scaled, y_train, y_test, rfe, rfe_selected,
        features, rf_grid, gb_grid)

    return {
        "kb_scores": kb_scores,
        "rfe_ranking": rfe_ranking,
        "rfe_selected": rfe_selected,
        "fi_rf": fi_rf,
        "poredjenje_svi_vs_top10": poredjenje,
    }
