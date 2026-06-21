# SAUSAU_Valorant_Projekat
Projekat iz predmeta Softverski algoritmi u sistemima automatskog upravljanja

Projekat "Анализа мете и предвиђање мечева на Valorant е-спорт турнирима." 
Mateja Stojišić RA180/2023

Ovaj repozitorijum sadrži:
  - **[DOKUMENTACIJA.md](DOKUMENTACIJA.md): Trenutno važeća, ažurirana dokumentacija projekta** (problem, dataset, metodologija, rezultati, identifikovani problemi i rješenja, sa svim grafovima)
  - Dokumentacija_projekta.docx: Raniji snapshot dokumentacije (prije refaktorisanja koda i deployment-a) - istorijski zapis, vidi DOKUMENTACIJA.md za trenutno stanje
  - Specifikacije projekta.docx: Unaprijed definisane specifikacije projekta (cilj)
  - folder kod: Pipeline za predikciju mečeva (treniranje + simulacija Masters London 2026) i analizu mete agenata, plus UI za korišćenje istreniranog modela
  - folder rezultati: Svi grafovi i istrenirani model (model_predictor.joblib) su smješteni ovdje

## Struktura `kod/` foldera

Pipeline za predikciju mečeva (nekad jedan london.py fajl od ~1400 linija) je
podijeljen po fazama:

  - `config.py` - putanje, konstante, lista timova/parova za Masters London 2026
  - `data_loading.py` - učitavanje CSV-ova i kanonizacija alias imena timova
  - `features.py` - feature engineering po timu (ELO, forma, eco stats, tier-1 uspjeh...)
  - `dataset_prep.py` - spajanje featura sa mečevima, diff-feature, train/test split
  - `eda.py` - korelaciona analiza i detekcija anomalija/outliera (IQR + boxplot)
  - `models.py` - treniranje 5 algoritama + voting ensemble, CV, GridSearchCV hiperparametri
  - `feature_selection.py` - SelectKBest, RFE, RF importance + poređenje svi-vs-top10 atributa
  - `predict.py` - `MatchPredictor` klasa (samostalan, picklable objekat za predikciju)
  - `simulate.py` - Swiss + double-elimination i Monte Carlo simulacija turnira
  - `main.py` - pokreće cijeli pipeline od početka do kraja, čuva istrenirani model
  - `agenti.py` - nezavisna analiza pick/win rate-a agenata 2021-2026 (9 grafova)
  - `app.py` - Streamlit UI za korišćenje istreniranog modela

## Postavljanje (kloniranje + dataset)

"Root projekta" niže u tekstu = folder u koji se repozitorijum klonira (onaj
koji direktno sadrži ovaj README.md, `kod/`, `rezultati/`, `requirements.txt`).

```
# 1. Kloniraj repo - kreirani folder SAUSAU_Valorant_Projekat/ JESTE root projekta
git clone https://github.com/pepsika/SAUSAU_Valorant_Projekat.git
cd SAUSAU_Valorant_Projekat

# 2. Dataset (~1.5GB, pojedini CSV-ovi preko 100MB) ne moze direktno u git,
# pa je dostupan kao asset uz GitHub Release. Repo je privatan, pa je
# potreban gh CLI (https://cli.github.com) ulogovan nalogom koji ima
# pristup ovom repozitorijumu (gh auth login)
gh release download dataset-v1 --repo pepsika/SAUSAU_Valorant_Projekat --pattern dataset.zip

# Raspakuj OVDJE, u root projekta (zip vec sadrzi top-level dataset/ folder,
# pa ce nakon ovoga postojati npr. SAUSAU_Valorant_Projekat/dataset/2021-2026/...)
python -m zipfile -e dataset.zip .
```

Sve putanje u `kod/config.py` su izvedene iz lokacije `config.py` (roditelj
`kod/` foldera = root projekta), tako da nakon raspakivanja pipeline radi bez
ikakvih izmjena bez obzira gdje je repozitorijum kloniran.

## Pokretanje

```
pip install -r requirements.txt

# 1. Treniranje (učitava podatke, trenira/tjuninguje modele, čuva model_predictor.joblib)
python kod/main.py

# 2. Analiza agenata (nezavisno od modela)
python kod/agenti.py

# 3. UI za predikciju mečeva (zahtijeva da je main.py već jednom pokrenut)
streamlit run kod/app.py
```
