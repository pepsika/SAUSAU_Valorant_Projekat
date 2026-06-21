"""Simulacija Masters London 2026: deterministicki (Swiss + double-elim sa
8 timova) i Monte Carlo format. Sve funkcije rade preko MatchPredictor objekta
(predict.MatchPredictor), tako da ne zavise od globalnog stanja modela."""

import os
import random

import matplotlib
matplotlib.use("Agg")  # neinteraktivni backend - skripta samo cuva fajlove (savefig), bez prozora
import matplotlib.pyplot as plt

from config import PATH_REZULTATI


def odigraj_mec(predictor, tim_a, tim_b, scores=None, log=True):
    pobjednik, vjer = predictor.predvidi_mec(tim_a, tim_b)
    gubitnik = tim_b if pobjednik == tim_a else tim_a
    if scores is not None:
        scores[pobjednik][0] += 1
        scores[gubitnik][1] += 1
    if log:
        print(f"  {tim_a:25s} vs {tim_b:25s} -> {pobjednik:25s} ({vjer:.1f}%)")
    return pobjednik, gubitnik


def simuliraj_swiss(predictor, timovi, r1_parovi=None, log=True):
    """Standardni Swiss format: 3 runde, do 2 pobjede ili 2 poraza.

    r1_parovi: lista (tim_a, tim_b) - STVARNI parovi iz drawa.
               Ako None, generise se zip(prva_polovina, druga_polovina).
    """
    scores = {t: [0, 0] for t in timovi}

    if log:
        print("\n--- SWISS ROUND 1 ---")
    if r1_parovi is None:
        polovina = len(timovi) // 2
        r1_parovi = list(zip(timovi[:polovina], timovi[polovina:]))
    for a, b in r1_parovi:
        odigraj_mec(predictor, a, b, scores, log)

    if log:
        print("\n--- SWISS ROUND 2 ---")
    t_1_0 = sorted([t for t, s in scores.items() if s == [1, 0]])
    t_0_1 = sorted([t for t, s in scores.items() if s == [0, 1]])
    for grupa in [t_1_0, t_0_1]:
        for i in range(0, len(grupa) - 1, 2):
            odigraj_mec(predictor, grupa[i], grupa[i + 1], scores, log)

    if log:
        print("\n--- SWISS ROUND 3 ---")
    t_2_0 = [t for t, s in scores.items() if s == [2, 0]]
    t_1_1 = sorted([t for t, s in scores.items() if s == [1, 1]])
    prolaz = list(t_2_0)
    for i in range(0, len(t_1_1) - 1, 2):
        pob, gub = odigraj_mec(predictor, t_1_1[i], t_1_1[i + 1], scores, log)
        prolaz.append(pob)
    return prolaz


def simuliraj_double_elim_8(predictor, direktni, swiss_survivori, log=True):
    """STVARAN format Masters London 2026:
    - 4 direktni (1. seedovi) BIRAJU protivnika iz 4 Swiss survivora
    - Redoslijed biranja je RANDOM
    - Svaki direktni bira tima koji mu daje najvecu sansu pobjede
    """
    if log:
        print("\n--- DRAW: 1. seedovi biraju protivnike (random redoslijed) ---")

    redoslijed_biranja = list(direktni)
    random.shuffle(redoslijed_biranja)

    dostupni = list(swiss_survivori)
    parovi_uqf = []
    for seed in redoslijed_biranja:
        protivnik = min(dostupni, key=lambda p: predictor.vjerovatnoca_pobjede(p, seed))
        dostupni.remove(protivnik)
        parovi_uqf.append((seed, protivnik))
        if log:
            p_seed = predictor.vjerovatnoca_pobjede(seed, protivnik) * 100
            print(f"  {seed:25s} bira {protivnik:25s} ({p_seed:.1f}% za seeda)")

    if log:
        print("\n--- UPPER QUARTERFINALS ---")
    uqf_pob, uqf_gub = [], []
    for a, b in parovi_uqf:
        p, g = odigraj_mec(predictor, a, b, log=log)
        uqf_pob.append(p); uqf_gub.append(g)

    if log:
        print("\n--- UPPER SEMIFINALS ---")
    usf_pob = []
    for i in range(0, len(uqf_pob), 2):
        p, _ = odigraj_mec(predictor, uqf_pob[i], uqf_pob[i + 1], log=log)
        usf_pob.append(p)
    uqf_gub_iz_sf = [t for pair in zip(uqf_pob[::2], uqf_pob[1::2])
                     for t in pair if t not in usf_pob]

    if log:
        print("\n--- LOWER ROUND 1 (UQF gubitnici)---")
    lr1_pob = []
    for i in range(0, len(uqf_gub), 2):
        p, _ = odigraj_mec(predictor, uqf_gub[i], uqf_gub[i + 1], log=log)
        lr1_pob.append(p)

    if log:
        print("\n--- LOWER ROUND 2 (UB SF gub vs LB R1 pob) ---")
    lr2_pob = []
    for i in range(len(lr1_pob)):
        p, _ = odigraj_mec(predictor, uqf_gub_iz_sf[i], lr1_pob[i], log=log)
        lr2_pob.append(p)

    if log:
        print("\n--- UPPER FINAL ---")
    uf_pob, uf_gub = odigraj_mec(predictor, usf_pob[0], usf_pob[1], log=log)

    if log:
        print("\n--- LOWER FINAL ---")
    lf_pob, _ = odigraj_mec(predictor, lr2_pob[0], lr2_pob[1] if len(lr2_pob) > 1 else uf_gub, log=log)

    if log:
        print("\n--- LOWER BRACKET FINAL ---")
    lbf_pob, _ = odigraj_mec(predictor, uf_gub, lf_pob, log=log)

    if log:
        print("\n--- GRAND FINAL ---")
    gf_pob, gf_gub = odigraj_mec(predictor, uf_pob, lbf_pob, log=log)
    return gf_pob


def simuliraj_masters_london(predictor, direktni_seedovi, swiss_timovi, r1_parovi, log=True):
    """Puna deterministicka simulacija turnira (Swiss + double-elim)."""
    print("\n" + "=" * 60)
    print("  MASTERS LONDON 2026 - DETERMINISTICKA PREDIKCIJA")
    print("=" * 60)

    swiss_prolaz = simuliraj_swiss(predictor, swiss_timovi, r1_parovi=r1_parovi, log=log)
    print(f"\nIz Swissa prolaze: {swiss_prolaz}")

    print(f"\nDirektni (1. seedovi): {direktni_seedovi}")
    print(f"Swiss survivori: {swiss_prolaz}")
    pobjednik = simuliraj_double_elim_8(predictor, direktni_seedovi, swiss_prolaz, log=log)

    print("\n" + "=" * 60)
    print(f"   POBJEDNIK (DETERMINISTICKI): {pobjednik}")
    print("=" * 60)
    return pobjednik


# ====================================================================
#  MONTE CARLO (pravi random, fokus na nesigurnost ishoda)
# ====================================================================

def _mc_odigraj(kes_proba, a, b, scores=None):
    p_a = kes_proba.get((a, b), 0.5)
    if random.random() < p_a:
        pob, gub = a, b
    else:
        pob, gub = b, a
    if scores is not None:
        if pob in scores: scores[pob][0] += 1
        if gub in scores: scores[gub][1] += 1
    return pob, gub


def _mc_swiss(kes_proba, timovi, r1_parovi=None):
    scores = {t: [0, 0] for t in timovi}

    if r1_parovi is None:
        polovina = len(timovi) // 2
        r1_parovi = list(zip(timovi[:polovina], timovi[polovina:]))
    for a, b in r1_parovi:
        _mc_odigraj(kes_proba, a, b, scores)

    t10 = [t for t, s in scores.items() if s == [1, 0]]
    t01 = [t for t, s in scores.items() if s == [0, 1]]
    random.shuffle(t10); random.shuffle(t01)
    for grupa in [t10, t01]:
        for i in range(0, len(grupa) - 1, 2):
            _mc_odigraj(kes_proba, grupa[i], grupa[i + 1], scores)

    t20 = [t for t, s in scores.items() if s == [2, 0]]
    t11 = [t for t, s in scores.items() if s == [1, 1]]
    random.shuffle(t11)
    prolaz = list(t20)
    for i in range(0, len(t11) - 1, 2):
        p, _ = _mc_odigraj(kes_proba, t11[i], t11[i + 1], scores)
        prolaz.append(p)
    return prolaz


def _mc_double_elim(kes_proba, direktni, swiss_survivori):
    redoslijed = list(direktni)
    random.shuffle(redoslijed)

    dostupni = list(swiss_survivori)
    parovi_uqf = []
    for seed in redoslijed:
        protivnik = min(dostupni, key=lambda p: kes_proba.get((p, seed), 0.5))
        dostupni.remove(protivnik)
        parovi_uqf.append((seed, protivnik))

    uqf_pob, uqf_gub = [], []
    for a, b in parovi_uqf:
        p, g = _mc_odigraj(kes_proba, a, b); uqf_pob.append(p); uqf_gub.append(g)

    usf_pob, usf_gub = [], []
    for i in range(0, len(uqf_pob), 2):
        p, g = _mc_odigraj(kes_proba, uqf_pob[i], uqf_pob[i + 1])
        usf_pob.append(p); usf_gub.append(g)

    lr1_pob = []
    for i in range(0, len(uqf_gub), 2):
        p, _ = _mc_odigraj(kes_proba, uqf_gub[i], uqf_gub[i + 1])
        lr1_pob.append(p)
    lr2_pob = [_mc_odigraj(kes_proba, usf_gub[i], lr1_pob[i])[0] for i in range(len(lr1_pob))]

    uf_pob, uf_gub = _mc_odigraj(kes_proba, usf_pob[0], usf_pob[1])
    if len(lr2_pob) >= 2:
        lf_pob, _ = _mc_odigraj(kes_proba, lr2_pob[0], lr2_pob[1])
    else:
        lf_pob = lr2_pob[0]
    lbf_pob, _ = _mc_odigraj(kes_proba, uf_gub, lf_pob)
    gf_pob, _ = _mc_odigraj(kes_proba, uf_pob, lbf_pob)
    return gf_pob


def monte_carlo(predictor, direktni_seedovi, swiss_timovi, r1_parovi, n_simulacija=10000):
    print("\n" + "=" * 60)
    print("  MONTE CARLO SIMULACIJA")
    print("=" * 60)

    masters_svi = direktni_seedovi + swiss_timovi

    print("Kesiranje vjerovatnoca...")
    kes_proba = {}
    for a in masters_svi:
        for b in masters_svi:
            if a != b:
                kes_proba[(a, b)] = predictor.vjerovatnoca_pobjede(a, b)

    pobjede = {}
    print(f"\nPokrecem {n_simulacija} simulacija...")
    for sim in range(n_simulacija):
        if sim % 2000 == 0 and sim > 0:
            print(f"  {sim}/{n_simulacija}...")
        sw_prolaz = _mc_swiss(kes_proba, list(swiss_timovi), r1_parovi=r1_parovi)
        winner = _mc_double_elim(kes_proba, direktni_seedovi, sw_prolaz)
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
