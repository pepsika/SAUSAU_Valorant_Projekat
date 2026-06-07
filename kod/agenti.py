"""
====================================================================
   VALORANT VCT - ANALIZA AGENATA (2021-2026)
====================================================================
Sadrzaj:
  1. Top agenti po prosjecnom pick rate (sve godine)
  2. Evolucija pick rate top 5 agenata kroz godine
  3. Win rate po agentu (iz teams_picked_agents)
  4. Meta promjene - agenti koji su porasli/pali
  5. Pick rate i win rate po mapama
  6. Top timovi po agentu (ko sta najcesce bira)
  7. Attacker vs Defender win rate po mapama
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

# ====================================================================
#  KONFIGURACIJA
# ====================================================================

PATH = r"C:\Users\KORISNIK\Desktop\6. semestar\Softverski algoritmi u sistemima automatskog upravljanja\Projekat\dataset"
PATH_REZULTATI = r"C:\Users\KORISNIK\Desktop\6. semestar\Softverski algoritmi u sistemima automatskog upravljanja\Projekat\rezultati"

GODINE = ["vct_2021", "vct_2022", "vct_2023", "vct_2024", "vct_2025", "vct_2026"]

# Stil grafova
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.color": "#dee2e6",
    "grid.alpha": 0.7,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BOJE_GODINA = {
    "vct_2021": "#e63946",
    "vct_2022": "#f4a261",
    "vct_2023": "#2a9d8f",
    "vct_2024": "#457b9d",
    "vct_2025": "#6a0572",
    "vct_2026": "#1d3557",
}

# ====================================================================
#  UCITAVANJE PODATAKA
# ====================================================================

print("=" * 60)
print("  UCITAVANJE PODATAKA")
print("=" * 60)

pick_files, team_files, maps_files = [], [], []

for godina in GODINE:
    base = os.path.join(PATH, "2021-2026", godina, "agents")

    fp = os.path.join(base, "agents_pick_rates.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df["godina"] = godina
        pick_files.append(df)

    fp = os.path.join(base, "teams_picked_agents.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df["godina"] = godina
        team_files.append(df)

    fp = os.path.join(base, "maps_stats.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df["godina"] = godina
        maps_files.append(df)

df_pick   = pd.concat(pick_files,  ignore_index=True)
df_teams  = pd.concat(team_files,  ignore_index=True)
df_maps   = pd.concat(maps_files,  ignore_index=True)

# Ciscenje pick rate kolone
df_pick["Pick Rate"] = pd.to_numeric(
    df_pick["Pick Rate"].astype(str).str.replace("%", "").str.strip(),
    errors="coerce"
)

# Win rate po agentu iz teams_picked_agents
df_teams["Total Wins By Map"]   = pd.to_numeric(df_teams["Total Wins By Map"],   errors="coerce")
df_teams["Total Maps Played"]   = pd.to_numeric(df_teams["Total Maps Played"],   errors="coerce")
df_teams["Total Loss By Map"]   = pd.to_numeric(df_teams["Total Loss By Map"],   errors="coerce")
df_teams["agent_win_rate"]      = (df_teams["Total Wins By Map"] /
                                   df_teams["Total Maps Played"]).replace([np.inf, -np.inf], np.nan)

# maps_stats - attacker/defender
df_maps["Attacker Side Win Percentage"] = pd.to_numeric(
    df_maps["Attacker Side Win Percentage"].astype(str).str.replace("%","").str.strip(),
    errors="coerce"
)
df_maps["Defender Side Win Percentage"] = pd.to_numeric(
    df_maps["Defender Side Win Percentage"].astype(str).str.replace("%","").str.strip(),
    errors="coerce"
)

print(f"  Pick rates:       {len(df_pick):,} redova")
print(f"  Teams/agenti:     {len(df_teams):,} redova")
print(f"  Maps stats:       {len(df_maps):,} redova")
print(f"  Jedinstveni agenti: {df_pick['Agent'].str.capitalize().nunique()}")
print(f"  Jedinstvene mape:   {df_teams['Map'].nunique()}")

# ====================================================================
#  1. TOP 10 AGENATA PO PROSJECNOM PICK RATE (SVE GODINE)
# ====================================================================

print("\n--- 1. Top agenti po pick rate (sve godine) ---")

# Filtriramo "All Maps" da izbjegnemo duplikate
pick_all = df_pick[df_pick["Map"] == "All Maps"].copy()
pick_all["Agent"] = pick_all["Agent"].str.capitalize()

top_pick = (pick_all.groupby("Agent")["Pick Rate"]
            .mean()
            .sort_values(ascending=False)
            .head(10))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_pick.index[::-1], top_pick.values[::-1],
               color=plt.cm.Blues(np.linspace(0.4, 0.85, 10)))
ax.set_xlabel("Prosjecni Pick Rate (%)", fontsize=12)
ax.set_title("Top 10 agenata po prosjecnom Pick Rate (2021-2026)", fontsize=14, fontweight="bold", pad=15)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())

for bar, val in zip(bars, top_pick.values[::-1]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "1_top_agenti_pick_rate.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 1_top_agenti_pick_rate.png")

# ====================================================================
#  2. EVOLUCIJA PICK RATE TOP 5 AGENATA KROZ GODINE
# ====================================================================

print("--- 2. Evolucija pick rate top 5 ---")

top5_agenti = top_pick.head(5).index.tolist()
df_evol = pick_all[pick_all["Agent"].isin(top5_agenti)]
pivot_evol = df_evol.groupby(["godina", "Agent"])["Pick Rate"].mean().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
for agent in top5_agenti:
    if agent in pivot_evol.columns:
        ax.plot(GODINE, [pivot_evol.loc[g, agent] if g in pivot_evol.index else 0 for g in GODINE],
                marker="o", linewidth=2.5, markersize=8, label=agent)

ax.set_title("Evolucija Pick Rate - Top 5 agenata (2021-2026)", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Prosjecni Pick Rate (%)", fontsize=12)
ax.set_xlabel("Sezona", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(fontsize=11, loc="upper left")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "2_evolucija_top5_agenata.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 2_evolucija_top5_agenata.png")

# ====================================================================
#  3. WIN RATE PO AGENTU (sve godine, min 20 mapa)
# ====================================================================

print("--- 3. Win rate po agentu ---")

df_teams["Agent"] = df_teams["Agent"].str.capitalize()

agent_wr = (df_teams.groupby("Agent")
            .agg(
                ukupne_pobjede=("Total Wins By Map", "sum"),
                ukupno_mapa=("Total Maps Played", "sum")
            )
            .reset_index())
agent_wr["win_rate"] = agent_wr["ukupne_pobjede"] / agent_wr["ukupno_mapa"] * 100
agent_wr = agent_wr[agent_wr["ukupno_mapa"] >= 20].sort_values("win_rate", ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))
boje = ["#2a9d8f" if wr >= 50 else "#e63946" for wr in agent_wr["win_rate"]]
bars = ax.bar(agent_wr["Agent"], agent_wr["win_rate"], color=boje, edgecolor="white", linewidth=0.5)
ax.axhline(50, color="black", linewidth=1.5, linestyle="--", alpha=0.6, label="50% (neutralno)")
ax.set_title("Win Rate po agentu (2021-2026, min 20 mapa)", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Win Rate (%)", fontsize=12)
ax.set_ylim(40, 65)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=9)

for bar, val in zip(bars, agent_wr["win_rate"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "3_win_rate_po_agentu.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 3_win_rate_po_agentu.png")

# ====================================================================
#  4. META PROMJENE - pick rate po godini (heatmapa)
# ====================================================================

print("--- 4. Meta promjene - heatmapa ---")

pick_all["Agent"] = pick_all["Agent"].str.capitalize()
pivot_meta = (pick_all.groupby(["godina", "Agent"])["Pick Rate"]
              .mean()
              .unstack(fill_value=0))

# Uzmi samo agente koji su imali bar 10% pick rate u nekoj godini
relevantni = pivot_meta.columns[(pivot_meta >= 10).any()]
pivot_meta = pivot_meta[relevantni]

fig, ax = plt.subplots(figsize=(16, 7))
sns.heatmap(pivot_meta.T, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Pick Rate (%)", "shrink": 0.8},
            ax=ax)
ax.set_title("Meta promjene po godini - Pick Rate agenata (%)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Sezona", fontsize=12)
ax.set_ylabel("Agent", fontsize=12)
plt.xticks(rotation=15)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "4_meta_promjene_heatmapa.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 4_meta_promjene_heatmapa.png")

# ====================================================================
#  5. PICK RATE I WIN RATE PO MAPAMA (bubble chart)
# ====================================================================

print("--- 5. Pick rate i win rate po mapama ---")

# Filtriramo samo specificne mape (ne "All Maps")
df_mape = df_teams[df_teams["Map"] != "All Maps"].copy()
df_mape["Agent"] = df_mape["Agent"].str.capitalize()

agent_mapa = (df_mape.groupby(["Map", "Agent"])
              .agg(
                  ukupne_pobjede=("Total Wins By Map", "sum"),
                  ukupno_mapa=("Total Maps Played", "sum")
              )
              .reset_index())
agent_mapa["win_rate"] = agent_mapa["ukupne_pobjede"] / agent_mapa["ukupno_mapa"] * 100
agent_mapa = agent_mapa[agent_mapa["ukupno_mapa"] >= 10]

# Pivot za heatmapu - top 15 agenata na mapama
top_agenti_mapa = (agent_mapa.groupby("Agent")["ukupno_mapa"]
                   .sum()
                   .nlargest(15)
                   .index.tolist())
agent_mapa_top = agent_mapa[agent_mapa["Agent"].isin(top_agenti_mapa)]

pivot_wr_mapa = agent_mapa_top.pivot_table(
    index="Agent", columns="Map", values="win_rate", aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot_wr_mapa, annot=True, fmt=".0f", cmap="RdYlGn",
            center=50, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Win Rate (%)", "shrink": 0.8},
            ax=ax)
ax.set_title("Win Rate agenata po mapama (%) - Top 15 agenata", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Mapa", fontsize=12)
ax.set_ylabel("Agent", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "5_win_rate_agenti_mape.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 5_win_rate_agenti_mape.png")

# ====================================================================
#  6. TOP 10 TIMOVA PO AGENTU (ko sta najcesce bira)
# ====================================================================

print("--- 6. Top timovi po agentu ---")

df_teams_all = df_teams[df_teams["Map"] == "All Maps"].copy() if "All Maps" in df_teams["Map"].values else df_teams.copy()
if "All Maps" not in df_teams["Map"].values:
    # Ako nema All Maps, agregiramo po timu i agentu
    df_teams_all = (df_teams.groupby(["Team", "Agent"])
                    .agg(Total_Maps=("Total Maps Played", "sum"))
                    .reset_index())
    df_teams_all.columns = ["Team", "Agent", "Total Maps Played"]
else:
    df_teams_all = df_teams_all[["Team", "Agent", "Total Maps Played"]].copy()

df_teams_all["Agent"] = df_teams_all["Agent"].str.capitalize()
df_teams_all["Total Maps Played"] = pd.to_numeric(df_teams_all["Total Maps Played"], errors="coerce")

# Top 6 agenata po ukupnom pick broju
top6_agenti_wr = (df_teams_all.groupby("Agent")["Total Maps Played"]
                  .sum()
                  .nlargest(6)
                  .index.tolist())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Top 10 timova po broju odigranih mapa sa agentom (2021-2026)",
             fontsize=14, fontweight="bold", y=1.02)

for ax, agent in zip(axes.flat, top6_agenti_wr):
    df_agent = (df_teams_all[df_teams_all["Agent"] == agent]
                .groupby("Team")["Total Maps Played"]
                .sum()
                .nlargest(10)
                .sort_values())
    if df_agent.empty:
        ax.set_visible(False)
        continue
    bars = ax.barh(df_agent.index, df_agent.values,
                   color=plt.cm.viridis(np.linspace(0.2, 0.9, len(df_agent))))
    ax.set_title(f"{agent}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Broj mapa", fontsize=9)
    for bar, val in zip(bars, df_agent.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                str(int(val)), va="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "6_top_timovi_po_agentu.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 6_top_timovi_po_agentu.png")

# ====================================================================
#  7. ATTACKER VS DEFENDER WIN RATE PO MAPAMA
# ====================================================================

print("--- 7. Attacker vs Defender win rate po mapama ---")

# Samo specificne mape (ne All Maps) i filtriraj nevazece
df_maps_spec = df_maps[
    (df_maps["Map"] != "All Maps") &
    (df_maps["Attacker Side Win Percentage"].notna()) &
    (df_maps["Defender Side Win Percentage"].notna())
].copy()

mapa_side = (df_maps_spec.groupby("Map")
             .agg(
                 att=("Attacker Side Win Percentage", "mean"),
                 def_=("Defender Side Win Percentage", "mean"),
                 odigrano=("Total Maps Played", "sum")
             )
             .reset_index())
mapa_side = mapa_side[mapa_side["odigrano"] >= 10].sort_values("att", ascending=False)

x = np.arange(len(mapa_side))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - width/2, mapa_side["att"],  width, label="Attacker", color="#e63946", alpha=0.85)
b2 = ax.bar(x + width/2, mapa_side["def_"], width, label="Defender", color="#457b9d", alpha=0.85)
ax.axhline(50, color="black", linewidth=1.2, linestyle="--", alpha=0.5, label="50%")

ax.set_title("Attacker vs Defender Win Rate po mapama (2021-2026)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Win Rate (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(mapa_side["Map"], rotation=20, ha="right", fontsize=10)
ax.set_ylim(35, 70)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(fontsize=11)

for bar, val in zip(list(b1) + list(b2),
                    list(mapa_side["att"]) + list(mapa_side["def_"])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "7_attacker_defender_mape.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 7_attacker_defender_mape.png")

# ====================================================================
#  8. PICK RATE VS WIN RATE SCATTER (koji agenti su OP?)
# ====================================================================

print("--- 8. Pick rate vs Win rate scatter ---")

# Spoji prosjecni pick rate i win rate po agentu
avg_pick = (pick_all.groupby("Agent")["Pick Rate"].mean().reset_index()
            .rename(columns={"Pick Rate": "avg_pick_rate"}))
avg_wr = agent_wr[["Agent", "win_rate", "ukupno_mapa"]].copy()

scatter_df = avg_pick.merge(avg_wr, on="Agent").dropna()
scatter_df = scatter_df[scatter_df["ukupno_mapa"] >= 30]

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    scatter_df["avg_pick_rate"],
    scatter_df["win_rate"],
    s=scatter_df["ukupno_mapa"] / 5,
    c=scatter_df["win_rate"],
    cmap="RdYlGn",
    vmin=45, vmax=60,
    alpha=0.8, edgecolors="white", linewidth=0.8
)

for _, row in scatter_df.iterrows():
    ax.annotate(row["Agent"],
                (row["avg_pick_rate"], row["win_rate"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.axhline(50, color="grey", linewidth=1, linestyle="--", alpha=0.6)
ax.axvline(scatter_df["avg_pick_rate"].mean(), color="grey",
           linewidth=1, linestyle="--", alpha=0.6)

ax.set_xlabel("Prosjecni Pick Rate (%)", fontsize=12)
ax.set_ylabel("Win Rate (%)", fontsize=12)
ax.set_title("Pick Rate vs Win Rate po agentu\n(velicina kruga = broj odigranih mapa)",
             fontsize=14, fontweight="bold", pad=15)

plt.colorbar(scatter, ax=ax, label="Win Rate (%)", shrink=0.8)

# Oznaci kvadrante
ax.text(0.98, 0.98, "Cesti + Uspjesni (OP?)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#2a9d8f", alpha=0.7)
ax.text(0.98, 0.02, "Cesti + Lose",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#e63946", alpha=0.7)
ax.text(0.02, 0.98, "Rijetki + Uspjesni (niche)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="#457b9d", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(PATH_REZULTATI, "8_pick_rate_vs_win_rate.png"), dpi=120)
plt.close()
print(f"  Sacuvano: 8_pick_rate_vs_win_rate.png")

# ====================================================================
#  9. AGENTI NA MASTERS LONDON 2026 (samo 2026 turniri)
# ====================================================================

print("--- 9. Agenti na Masters London 2026 ---")

df_2026 = pick_all[pick_all["godina"] == "vct_2026"].copy()
if not df_2026.empty:
    top_2026 = (df_2026.groupby("Agent")["Pick Rate"]
                .mean()
                .sort_values(ascending=False)
                .head(15))

    fig, ax = plt.subplots(figsize=(12, 6))
    boje_2026 = plt.cm.plasma(np.linspace(0.2, 0.85, len(top_2026)))
    bars = ax.barh(top_2026.index[::-1], top_2026.values[::-1],
                   color=boje_2026)
    ax.set_xlabel("Prosjecni Pick Rate (%)", fontsize=12)
    ax.set_title("Top 15 agenata - VCT 2026 (aktuelna meta)", fontsize=14,
                 fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, top_2026.values[::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_REZULTATI, "9_agenti_2026_aktuelna_meta.png"), dpi=120)
    plt.close()
    print(f"  Sacuvano: 9_agenti_2026_aktuelna_meta.png")

# ====================================================================
#  PRINTANJE KLJUCNIH STATISTIKA
# ====================================================================

print("\n" + "=" * 60)
print("  KLJUCNE STATISTIKE")
print("=" * 60)

print("\nTop 10 agenata po pick rate (sve godine):")
for agent, pr in top_pick.items():
    print(f"  {agent:15s}: {pr:.1f}%")

print("\nTop 10 agenata po win rate (min 20 mapa):")
for _, row in agent_wr.head(10).iterrows():
    print(f"  {row['Agent']:15s}: {row['win_rate']:.1f}%  ({int(row['ukupno_mapa'])} mapa)")

print("\nNajveci rast pick rate-a (2021 -> 2026):")
if "vct_2021" in pivot_meta.index and "vct_2026" in pivot_meta.index:
    rast = (pivot_meta.loc["vct_2026"] - pivot_meta.loc["vct_2021"]).sort_values(ascending=False)
    for agent, delta in rast.head(5).items():
        print(f"  {agent:15s}: {delta:+.1f}%")
    print("\nNajveci pad pick rate-a (2021 -> 2026):")
    for agent, delta in rast.tail(5).items():
        print(f"  {agent:15s}: {delta:+.1f}%")

print("\nMape: Attacker-friendly (>50% ATT win rate):")
att_friendly = mapa_side[mapa_side["att"] > 50].sort_values("att", ascending=False)
for _, row in att_friendly.iterrows():
    print(f"  {row['Map']:15s}: ATT {row['att']:.1f}% | DEF {row['def_']:.1f}%")

print("\nMape: Defender-friendly (>50% DEF win rate):")
def_friendly = mapa_side[mapa_side["def_"] > 50].sort_values("def_", ascending=False)
for _, row in def_friendly.iterrows():
    print(f"  {row['Map']:15s}: DEF {row['def_']:.1f}% | ATT {row['att']:.1f}%")

print("\n" + "=" * 60)
print(f"  GOTOVO! Svi grafovi sacuvani u: {PATH_REZULTATI}")
print("=" * 60)