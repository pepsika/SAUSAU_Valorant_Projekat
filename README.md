# SAUSAU_Valorant_Projekat
Projekat iz predmeta Softverski algoritmi u sistemima automatskog upravljanja

Projekat "Анализа мете и предвиђање мечева на Valorant е-спорт турнирима." 
Mateja Stojišić RA180/2023

Ovaj repozitorijum sadrži dva dokumenta i tri foldera:
  - Dokumentacija_projekta.docx: Za pročitati dokle je postignuto sa projektom samim
  - Specifikacije projekta.docx: Unaprijed definisane specifikacije projekta (cilj)
  - folder dataset: Sadrži kompletan dataset korišten za treniranje i testiranje ML modela
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

