"""Streamlit UI za predikciju pobjednika Valorant VCT meceva.

Pokretanje:  streamlit run app.py
Zahtijeva da je main.py vec pokrenut bar jednom (kreira model_predictor.joblib
u rezultati/ folderu preko MatchPredictor.izgradi_red_za_predikciju pipeline-a).
"""

import os

import joblib
import streamlit as st

from config import PATH_MODEL

st.set_page_config(page_title="VCT Masters London 2026 - Predikcija", page_icon="🎯")


@st.cache_resource
def ucitaj_model():
    return joblib.load(PATH_MODEL)


st.title("Valorant VCT - Predikcija pobjednika meca")

if not os.path.exists(PATH_MODEL):
    st.error(
        f"Model nije pronadjen na: {PATH_MODEL}\n\n"
        "Prvo pokreni `python main.py` da se model istrenira i sacuva."
    )
    st.stop()

predictor = ucitaj_model()
st.caption(f"Ucitan model: **{predictor.model_name}**")

timovi = predictor.list_teams()

col1, col2 = st.columns(2)
with col1:
    tim_a = st.selectbox("Tim A", timovi, index=0)
with col2:
    default_b = 1 if len(timovi) > 1 else 0
    tim_b = st.selectbox("Tim B", timovi, index=default_b)

if st.button("Predvidi pobjednika", type="primary"):
    if tim_a == tim_b:
        st.warning("Izaberi dva razlicita tima.")
    else:
        pobjednik, vjerovatnoca = predictor.predvidi_mec(tim_a, tim_b)
        p_a = predictor.vjerovatnoca_pobjede(tim_a, tim_b) * 100
        p_b = 100 - p_a

        st.subheader(f"Pobjednik: {pobjednik} ({vjerovatnoca:.1f}%)")
        st.progress(p_a / 100, text=f"{tim_a}: {p_a:.1f}%")
        st.progress(p_b / 100, text=f"{tim_b}: {p_b:.1f}%")

st.divider()
st.caption(
    "Model je treniran na VCT mecevima 2021-2026 (ELO, forma, eco statistike, "
    "tier-1 uspjeh, Stage 1 2026 standings). Vidi kod/main.py za detalje treninga."
)
