"""Treniranje modela, unakrsna validacija, podesavanje hiperparametara.

Pored accuracy (glavna metrika za poredjenje), racunaju se i ROC-AUC i Brier
score - korisne dopunske metrike za binarnu klasifikaciju sa vjerovatnocama
(ROC-AUC mjeri razdvajanje klasa nezavisno od praga, Brier mjeri kalibraciju
vjerovatnoca - bitno jer se predict_proba koristi za Monte Carlo simulaciju).
Napomena: R^2 (koeficijent determinacije) je metrika za regresiju i ovdje se
namjerno ne koristi - cilj ovog modela je klasifikacija (pobjednik meca).
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, brier_score_loss,
)

from config import PATH_REZULTATI, RANDOM_STATE


def build_models():
    return {
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


def _fit_sa_tezinama(model, X_train, y_train, sample_weights):
    try:
        if isinstance(model, Pipeline):
            model.fit(X_train, y_train, clf__sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weights)
    except Exception:
        model.fit(X_train, y_train)
    return model


def train_models(X_train, y_train, X_test, y_test, sample_weights):
    """Trenira sve osnovne modele + voting ensemble, vraca metrike i najbolji model."""
    print("\n" + "=" * 60)
    print("  TRENIRANJE MODELA")
    print("=" * 60)

    modeli = build_models()
    rezultati_test = {}
    metrike = {}
    for naziv, model in modeli.items():
        _fit_sa_tezinama(model, X_train, y_train, sample_weights)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        rezultati_test[naziv] = acc
        metrike[naziv] = {"accuracy": acc, "roc_auc": auc, "brier": brier}
        print(f"  {naziv:25s}: acc={acc:.3f}  ROC-AUC={auc:.3f}  Brier={brier:.3f}")

    print("\n  Treniram VOTING ENSEMBLE (LR + SVM + GB)...")
    ensemble = VotingClassifier(
        estimators=[
            ("lr", modeli["Logisticka regresija"]),
            ("svm", modeli["SVM (RBF)"]),
            ("gb", modeli["Gradient Boosting"]),
        ],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    y_proba_ens = ensemble.predict_proba(X_test)[:, 1]
    acc_ens = accuracy_score(y_test, y_pred_ens)
    auc_ens = roc_auc_score(y_test, y_proba_ens)
    brier_ens = brier_score_loss(y_test, y_proba_ens)
    rezultati_test["Voting Ensemble"] = acc_ens
    metrike["Voting Ensemble"] = {"accuracy": acc_ens, "roc_auc": auc_ens, "brier": brier_ens}
    modeli["Voting Ensemble"] = ensemble
    print(f"  {'Voting Ensemble':25s}: acc={acc_ens:.3f}  ROC-AUC={auc_ens:.3f}  Brier={brier_ens:.3f}")

    najbolji_naziv = max(rezultati_test, key=rezultati_test.get)
    print(f"\n=> Najbolji: {najbolji_naziv} ({rezultati_test[najbolji_naziv]:.3f})")
    najbolji_model = modeli[najbolji_naziv]

    print(f"\nClassification report ({najbolji_naziv}):")
    print(classification_report(y_test, najbolji_model.predict(X_test),
                                 target_names=["Team B won", "Team A won"]))

    return {
        "modeli": modeli,
        "rezultati_test": rezultati_test,
        "metrike": metrike,
        "najbolji_naziv": najbolji_naziv,
        "najbolji_model": najbolji_model,
    }


def cross_validate_models(modeli, X_train, y_train):
    """K-fold CV (k=5) SAMO na trening setu - izbjegava data leakage iz test seta."""
    print("\n" + "=" * 60)
    print("  K-FOLD CROSS VALIDACIJA (k=5, samo TRAIN)")
    print("=" * 60)

    cv_rezultati = {}
    for naziv, model in modeli.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_rezultati[naziv] = scores
        print(f"  {naziv:25s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    plt.figure(figsize=(10, 5))
    plt.boxplot(cv_rezultati.values(), labels=cv_rezultati.keys())
    plt.title("K-Fold CV (5-fold) na trening setu")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_REZULTATI, "poredenje_algoritama_v2.png"), dpi=100)
    plt.close()

    return cv_rezultati


def plot_confusion_matrix(najbolji_model, najbolji_naziv, X_test, y_test):
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
    return cm


def plot_feature_importance(najbolji_model, najbolji_naziv, features):
    if not hasattr(najbolji_model, "feature_importances_"):
        return None

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
    return fi


def tune_hyperparameters(X_train, y_train, X_test, y_test, sample_weights, rezultati_test,
                          najbolji_model, najbolji_naziv):
    """GridSearchCV za svaki algoritam; vraca (eventualno azuriran) najbolji model."""
    print("\n" + "=" * 60)
    print("  PODESAVANJE HIPERPARAMETARA (GridSearchCV)")
    print("=" * 60)

    print("\n  Logisticka regresija...")
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    lr_params = {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1", "l2"],
                 "clf__solver": ["liblinear"]}
    lr_grid = GridSearchCV(lr_pipe, lr_params, cv=5, scoring="accuracy", n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    print(f"    Najbolji parametri: {lr_grid.best_params_}")
    print(f"    CV accuracy:  {lr_grid.best_score_:.3f}")
    print(f"    Test accuracy: {lr_grid.score(X_test, y_test):.3f}")

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

    print("\n  SVM (RBF)...")
    svm_pipe = Pipeline([("scaler", StandardScaler()),
                          ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])
    svm_params = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", "auto", 0.01, 0.1]}
    svm_grid = GridSearchCV(svm_pipe, svm_params, cv=5, scoring="accuracy", n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    print(f"    Najbolji parametri: {svm_grid.best_params_}")
    print(f"    CV accuracy:  {svm_grid.best_score_:.3f}")
    print(f"    Test accuracy: {svm_grid.score(X_test, y_test):.3f}")

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

    print("\n  === POREDENJE: DEFAULT vs TUNED ===")
    tuned_modeli = {
        "Logisticka regresija": lr_grid,
        "Random Forest": rf_grid,
        "Gradient Boosting": gb_grid,
        "SVM (RBF)": svm_grid,
        "KNN": knn_grid,
    }
    print(f"  {'Algoritam':25s} {'Default':>10s} {'Tuned':>10s} {'Razlika':>10s}")
    print(f"  {'-' * 55}")
    for naziv, grid in tuned_modeli.items():
        default_acc = rezultati_test.get(naziv, rezultati_test.get("KNN (k=7)", 0))
        tuned_acc = grid.score(X_test, y_test)
        diff = tuned_acc - default_acc
        print(f"  {naziv:25s} {default_acc:10.3f} {tuned_acc:10.3f} {diff:+10.3f}")

    sve_tuned = {n: g.score(X_test, y_test) for n, g in tuned_modeli.items()}
    best_tuned_naziv = max(sve_tuned, key=sve_tuned.get)
    best_tuned_acc = sve_tuned[best_tuned_naziv]

    if best_tuned_acc > rezultati_test[najbolji_naziv]:
        print(f"\n  Tuned '{best_tuned_naziv}' ({best_tuned_acc:.3f}) je bolji od "
              f"default '{najbolji_naziv}' ({rezultati_test[najbolji_naziv]:.3f})")
        najbolji_model = tuned_modeli[best_tuned_naziv].best_estimator_
        najbolji_naziv = f"{best_tuned_naziv} (tuned)"
        print("  -> Koristim tuned model za predikcije")
    else:
        print(f"\n  Default '{najbolji_naziv}' je i dalje najbolji ({rezultati_test[najbolji_naziv]:.3f})")

    return {
        "tuned_modeli": tuned_modeli,
        "najbolji_model": najbolji_model,
        "najbolji_naziv": najbolji_naziv,
        "lr_grid": lr_grid, "rf_grid": rf_grid, "gb_grid": gb_grid,
        "svm_grid": svm_grid, "knn_grid": knn_grid,
    }
