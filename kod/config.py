"""Konfiguracija i konstante zajednicke svim modulima."""

import os

PATH = r"C:\Users\KORISNIK\Desktop\6. semestar\Softverski algoritmi u sistemima automatskog upravljanja\Projekat\dataset"
PATH_REZULTATI = r"C:\Users\KORISNIK\Desktop\6. semestar\Softverski algoritmi u sistemima automatskog upravljanja\Projekat\rezultati"
PATH_MODEL = os.path.join(PATH_REZULTATI, "model_predictor.joblib")

GODINE = ["vct_2021", "vct_2022", "vct_2023", "vct_2024", "vct_2025", "vct_2026"]

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

# Recent (2025/2026) medjunarodni turniri - za "trenutnu formu na top sceni"
RECENT_INT_TURNIRI = [
    "Valorant Champions 2025", "Valorant Masters Bangkok 2025",
    "Valorant Masters Toronto 2025", "Valorant Masters Santiago 2026",
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

# Masters London 2026 - sastav turnira (iz Stage 1 standings)
DIREKTNI_SEEDOVI = ["G2 Esports", "Team Heretics", "Paper Rex", "EDward Gaming"]
SWISS_TIMOVI = [
    "Xi Lai Gaming", "NRG", "Team Vitality", "Dragon Ranger Gaming",
    "FULL SENSE", "FUT Esports", "LEVIATÁN", "Global Esports",
]

# Stvarni parovi Round 1 (sa vlr.gg, draw 25. maj 2026)
# Bitno: izvuceni nakon VCT Americas Stage 1 finala, nisu generisani sortiranjem
# - moraju biti hardcode-ovani. Tek od Round 2 nadalje parovi se generisu
# dinamicki po Swiss formatu (1-0 vs 1-0, 0-1 vs 0-1).
STVARNI_R1_PAROVI = [
    ("Xi Lai Gaming", "NRG"),
    ("Team Vitality", "Dragon Ranger Gaming"),
    ("FULL SENSE", "FUT Esports"),
    ("LEVIATÁN", "Global Esports"),
]
