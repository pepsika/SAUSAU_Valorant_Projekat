"""MatchPredictor - samostalan, picklable objekat koji zna predvidjeti
ishod meca izmedju dva tima. Pakuje model + sve podatke koje predikcija
treba (team featuri, istorija meceva za H2H/legacy win rate), tako da se
moze sacuvati sa joblib.dump() i kasnije ucitati u UI/API bez ponovnog
ucitavanja cijelog dataseta - to je artefakt za 'Deployment modela' fazu.
"""

import numpy as np


class MatchPredictor:
    def __init__(self, model, model_name, features, team_features, df_pro, alias_mapping):
        self.model = model
        self.model_name = model_name
        self.features = list(features)
        self.alias_mapping = dict(alias_mapping)

        self.team_feat_dict = team_features.set_index("Team").to_dict("index")
        self.feat_default = {
            kol: team_features[kol].median()
            for kol in team_features.columns if kol != "Team"
        }

        kolone = ["Team A", "Team B", "team_a_won",
                  "win_rate_a", "win_rate_b", "broj_meceva_a", "broj_meceva_b"]
        kolone = [k for k in kolone if k in df_pro.columns]
        self.match_history = df_pro[kolone].reset_index(drop=True)

        self.known_teams = sorted(team_features["Team"].dropna().unique().tolist())

    def list_teams(self):
        return list(self.known_teams)

    def normalizuj_ime(self, tim):
        # Kanonizacija je vec uradjena na treningu - ovo je backup za stare aliase
        return self.alias_mapping.get(tim, tim)

    def dohvati_team_features(self, tim_naziv):
        tim = self.normalizuj_ime(tim_naziv)
        if tim in self.team_feat_dict:
            return self.team_feat_dict[tim]
        # Pokusaj parcijalni match (npr. razlicit capitalization/spacing)
        for t in self.team_feat_dict:
            if tim.lower() in str(t).lower() or str(t).lower() in tim.lower():
                return self.team_feat_dict[t]
        return self.feat_default

    def dohvati_win_rate_iz_pro(self, tim_naziv):
        """Vraca posljednji poznati legacy win rate i broj meceva za tim."""
        tim = self.normalizuj_ime(tim_naziv)
        df = self.match_history
        kao_a = df[df["Team A"] == tim][["win_rate_a", "broj_meceva_a"]].tail(1)
        kao_b = df[df["Team B"] == tim][["win_rate_b", "broj_meceva_b"]].tail(1)
        if not kao_a.empty:
            return float(kao_a["win_rate_a"].values[0]), int(kao_a["broj_meceva_a"].values[0])
        if not kao_b.empty:
            return float(kao_b["win_rate_b"].values[0]), int(kao_b["broj_meceva_b"].values[0])
        return 0.5, 0

    def izgradi_red_za_predikciju(self, tim_a, tim_b):
        """Vraca feature-red (1, n_features) za par tim_a vs tim_b."""
        wr_a, mec_a = self.dohvati_win_rate_iz_pro(tim_a)
        wr_b, mec_b = self.dohvati_win_rate_iz_pro(tim_b)

        tim_a_n = self.normalizuj_ime(tim_a)
        tim_b_n = self.normalizuj_ime(tim_b)
        df = self.match_history
        h2h = df[
            ((df["Team A"] == tim_a_n) & (df["Team B"] == tim_b_n)) |
            ((df["Team A"] == tim_b_n) & (df["Team B"] == tim_a_n))
        ]
        h2h_rate = 0.5
        if len(h2h) > 0:
            pob_a = len(h2h[
                ((h2h["Team A"] == tim_a_n) & (h2h["team_a_won"] == 1)) |
                ((h2h["Team B"] == tim_a_n) & (h2h["team_a_won"] == 0))
            ])
            h2h_rate = pob_a / len(h2h)

        feat_a = self.dohvati_team_features(tim_a)
        feat_b = self.dohvati_team_features(tim_b)

        red = {
            "h2h_win_rate_a": h2h_rate,
            "h2h_mecevi": len(h2h),
            "a_elo": feat_a.get("elo", 1500),
            "b_elo": feat_b.get("elo", 1500),
            "diff_elo": feat_a.get("elo", 1500) - feat_b.get("elo", 1500),
            "a_forma_15": feat_a.get("forma_15", 0.5),
            "b_forma_15": feat_b.get("forma_15", 0.5),
            "diff_forma": feat_a.get("forma_15", 0.5) - feat_b.get("forma_15", 0.5),
            "a_dynamic_wr": feat_a.get("dynamic_wr", 0.5),
            "b_dynamic_wr": feat_b.get("dynamic_wr", 0.5),
            "diff_dynamic_wr": feat_a.get("dynamic_wr", 0.5) - feat_b.get("dynamic_wr", 0.5),
            "diff_dynamic_mecevi": feat_a.get("dynamic_mecevi", 0) - feat_b.get("dynamic_mecevi", 0),
            "a_tier1_wr": feat_a.get("tier1_wr", 0.3),
            "b_tier1_wr": feat_b.get("tier1_wr", 0.3),
            "diff_tier1_wr": feat_a.get("tier1_wr", 0.3) - feat_b.get("tier1_wr", 0.3),
            "diff_tier1_iskustvo": feat_a.get("tier1_meceva", 0) - feat_b.get("tier1_meceva", 0),
            "a_recent_int_wr": feat_a.get("recent_int_wr", 0.3),
            "b_recent_int_wr": feat_b.get("recent_int_wr", 0.3),
            "diff_recent_int_wr": feat_a.get("recent_int_wr", 0.3) - feat_b.get("recent_int_wr", 0.3),
            "diff_champ_pts": feat_a.get("championship_points", 0) - feat_b.get("championship_points", 0),
            "diff_stage1": feat_a.get("stage1_score", 0) - feat_b.get("stage1_score", 0),
            "diff_map_wr": feat_a.get("map_wr", 0.5) - feat_b.get("map_wr", 0.5),
            "diff_rating": feat_a.get("rating_avg", 1.0) - feat_b.get("rating_avg", 1.0),
            "diff_acs": feat_a.get("acs_avg", 200) - feat_b.get("acs_avg", 200),
            "diff_adr": feat_a.get("adr_avg", 130) - feat_b.get("adr_avg", 130),
            "diff_pistol": feat_a.get("pistol_wr", 0.5) - feat_b.get("pistol_wr", 0.5),
            "diff_full_buy": feat_a.get("full_buy_wr", 0.5) - feat_b.get("full_buy_wr", 0.5),
            "diff_eco": feat_a.get("eco_wr", 0.5) - feat_b.get("eco_wr", 0.5),
            "diff_kast": feat_a.get("kast_avg", 70) - feat_b.get("kast_avg", 70),
            "diff_fkpr": feat_a.get("fkpr_avg", 0.1) - feat_b.get("fkpr_avg", 0.1),
            "diff_clutch": feat_a.get("clutch_avg", 0.15) - feat_b.get("clutch_avg", 0.15),
            "diff_hs": feat_a.get("hs_avg", 25) - feat_b.get("hs_avg", 25),
            "win_rate_a": wr_a, "win_rate_b": wr_b,
            "diff_win_rate": wr_a - wr_b,
        }
        return np.array([[red.get(f, 0.0) for f in self.features]])

    def vjerovatnoca_pobjede(self, tim_a, tim_b):
        """Vraca vjerovatnocu da tim_a pobijedi."""
        X = self.izgradi_red_za_predikciju(tim_a, tim_b)
        proba = self.model.predict_proba(X)[0]
        return float(proba[1])

    def predvidi_mec(self, tim_a, tim_b):
        p_a = self.vjerovatnoca_pobjede(tim_a, tim_b)
        if p_a >= 0.5:
            return tim_a, p_a * 100
        return tim_b, (1 - p_a) * 100
