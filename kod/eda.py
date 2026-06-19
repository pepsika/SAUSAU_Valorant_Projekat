"""Eksplorativna analiza skupa: korelaciona analiza i detekcija anomalija/outliera."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATH_REZULTATI


def correlation_analysis(X_train, features):
    """Korelaciona matrica train featura + parovi sa jakom korelacijom (|r| > 0.7)."""
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

    df_corr = pd.DataFrame(X_train, columns=features)
    corr_matrix = df_corr.corr()

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

    print("\nParovi atributa sa jakom korelacijom (|r| > 0.7):")
    jaki_parovi = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
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

    return corr_matrix, jaki_parovi


def outlier_analysis(X_train, features, iqr_k=1.5):
    """Detekcija ekstremnih/cudnih vrijednosti po atributu (IQR metoda) + boxplotovi.

    Ne uklanja outliere automatski - kod ovog problema ekstremne vrijednosti
    (npr. tim sa ELO 1900 ili 0% pistol win rate) su CESTO stvaran signal
    (dominantan/slab tim), ne greska u podacima. Ova analiza ih samo prijavljuje
    da bi se mogla napraviti informisana odluka.
    """
    print("\n" + "=" * 60)
    print("  DETEKCIJA ANOMALIJA / EKSTREMNIH VRIJEDNOSTI (IQR metoda)")
    print("=" * 60)

    df = pd.DataFrame(X_train, columns=features)
    izvjestaj = []
    for kol in df.columns:
        q1, q3 = df[kol].quantile(0.25), df[kol].quantile(0.75)
        iqr = q3 - q1
        donja, gornja = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        broj_outliera = int(((df[kol] < donja) | (df[kol] > gornja)).sum())
        izvjestaj.append({
            "feature": kol, "broj_outliera": broj_outliera,
            "procenat": broj_outliera / len(df) * 100,
            "donja_granica": donja, "gornja_granica": gornja,
        })

    izvjestaj_df = pd.DataFrame(izvjestaj).sort_values("broj_outliera", ascending=False)
    print("\nTop 10 atributa po broju outliera (IQR k=1.5):")
    for _, row in izvjestaj_df.head(10).iterrows():
        print(f"  {row['feature']:25s}  {row['broj_outliera']:4d} ({row['procenat']:.1f}%)  "
              f"granice=[{row['donja_granica']:.2f}, {row['gornja_granica']:.2f}]")

    # Boxplotovi za atribute sa najvise outliera (vizualna provjera)
    top_kol = izvjestaj_df.head(12)["feature"].tolist()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, kol in zip(axes.flat, top_kol):
        ax.boxplot(df[kol].dropna(), vert=True)
        ax.set_title(kol, fontsize=9)
    for ax in axes.flat[len(top_kol):]:
        ax.set_visible(False)
    plt.suptitle("Boxplot - atributi sa najvise ekstremnih vrijednosti", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_REZULTATI, "outlieri_boxplot.png"), dpi=120)
    plt.close()
    print("\n  Sacuvano: outlieri_boxplot.png")

    return izvjestaj_df
