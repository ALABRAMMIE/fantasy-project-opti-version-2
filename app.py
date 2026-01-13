import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
from io import BytesIO
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(page_title="NBA Team Optimizer v2", layout="wide")
st.title("🏀 NBA Team Optimizer v2")
st.markdown("""
Deze tool optimaliseert NBA team-selecties door wedstrijden te simuleren.
Het koppelt tegenstanders via **GameID** en bepaalt de winnaar op basis van jouw ingestelde **Tiers**.
""")

# ==========================================
# 1. SIDEBAR: INSTELLINGEN
# ==========================================

st.sidebar.header("⚙️ Solver Settings")

# Budget & Constraints
budget = st.sidebar.number_input("Max Budget", value=100.0, step=0.5)
team_size = st.sidebar.number_input("Aantal Teams te kiezen", min_value=1, value=5, step=1)
num_lineups = st.sidebar.number_input("Aantal Lineups genereren", min_value=1, max_value=50, value=5)
min_diff = st.sidebar.number_input("Minimaal verschil (aantal teams)", min_value=0, max_value=team_size, value=1)

# GameID Constraint
avoid_opposing = st.sidebar.checkbox("Max 1 team per wedstrijd kiezen", value=True, help="Voorkomt dat je beide teams uit dezelfde GameID kiest.")

st.sidebar.markdown("---")
st.sidebar.header("🎲 Win Probability Tiers")
st.sidebar.info("Bepaal hier de winstkans per Tier. De simulatie gebruikt deze kansen om per wedstrijd een winnaar aan te wijzen.")

# Standaard configuratie
default_tier_data = {
    "Tier": [1, 2, 3, 4, 5],
    "Label": ["Heavy Favorite", "Favorite", "Toss Up", "Underdog", "Longshot"],
    "Win %": [90, 70, 50, 30, 10],      # Relatieve sterkte
    "Win Bonus": [5.0, 5.0, 5.0, 5.0, 5.0]  # Punten erbij als ze winnen
}

df_tier_config = pd.DataFrame(default_tier_data)

# Bewerkbare tabel
edited_tiers = st.sidebar.data_editor(
    df_tier_config,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Tier": st.column_config.NumberColumn("Tier ID", format="%d"),
        "Label": st.column_config.TextColumn("Omschrijving"),
        "Win %": st.column_config.NumberColumn("Kracht / Win %", min_value=1, max_value=100),
        "Win Bonus": st.column_config.NumberColumn("Bonus Pts", min_value=0.0, format="%.1f")
    },
    key="tier_editor"
)

# Settings opslaan in dictionary
tier_settings = {}
for index, row in edited_tiers.iterrows():
    try:
        t_id = int(row["Tier"])
        p_win = float(row["Win %"])
        bonus = float(row["Win Bonus"])
        tier_settings[t_id] = {"prob": p_win, "bonus": bonus}
    except:
        continue

# ==========================================
# 2. FILE UPLOAD & VALIDATION
# ==========================================

st.markdown("### 1. Upload Teams Data")
uploaded_file = st.file_uploader("Upload Excel bestand", type=["xlsx"])

if not uploaded_file:
    st.info("Upload een Excel bestand met kolommen: Name, Value, FTPS, OutcomeTier, GameID")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Fout bij lezen bestand: {e}")
    st.stop()

# Check verplichte kolommen
required_cols = {"Name", "Value", "OutcomeTier"}
if not required_cols.issubset(df.columns):
    st.error(f"❌ Het bestand mist: {required_cols - set(df.columns)}")
    st.stop()

# GameID is optioneel, maar nodig voor brackets
if "GameID" not in df.columns:
    st.warning("⚠️ Geen 'GameID' kolom gevonden. Wedstrijd-koppeling werkt niet (simulatie wordt volledig random).")
    df["GameID"] = df.index # Fake ID zodat code niet crasht

if "FTPS" not in df.columns:
    df["FTPS"] = 0.0

# Data types
df["OutcomeTier"] = pd.to_numeric(df["OutcomeTier"], errors='coerce').fillna(3).astype(int)
nba_teams = df.to_dict("records")

with st.expander("🔍 Bekijk geüploade data"):
    st.dataframe(df)

# ==========================================
# 3. SIMULATION LOGIC (COUPLED)
# ==========================================

def run_simulation_for_all_games(teams_data):
    """
    Simuleert alle wedstrijden gebaseerd op GameID.
    Geeft terug: Dict {TeamNaam: TotalePunten}
    """
    simulated_scores = {}
    simulated_outcomes = {}
    
    # 1. Groepeer teams per GameID
    games = defaultdict(list)
    for t in teams_data:
        gid = t.get("GameID")
        if pd.notna(gid):
            games[gid].append(t)
        else:
            # Geen GameID? Behandel als solo team (50% kans of tier based)
            games[f"solo_{t['Name']}"].append(t)

    # 2. Speel de wedstrijden
    for gid, opponents in games.items():
        if len(opponents) == 2:
            # HET ECHTE WERK: Team A vs Team B
            tA = opponents[0]
            tB = opponents[1]
            
            settA = tier_settings.get(tA["OutcomeTier"], {"prob": 50, "bonus": 0})
            settB = tier_settings.get(tB["OutcomeTier"], {"prob": 50, "bonus": 0})
            
            # Bereken relatieve kans: A / (A + B)
            total_prob = settA["prob"] + settB["prob"]
            if total_prob == 0: total_prob = 1
            
            prob_A_wins = settA["prob"] / total_prob
            
            # Gooi dobbelsteen
            if random.random() < prob_A_wins:
                # A Wint
                simulated_scores[tA["Name"]] = tA.get("FTPS", 0) + settA["bonus"]
                simulated_scores[tB["Name"]] = tB.get("FTPS", 0)
                simulated_outcomes[tA["Name"]] = "WIN"
                simulated_outcomes[tB["Name"]] = "LOSS"
            else:
                # B Wint
                simulated_scores[tA["Name"]] = tA.get("FTPS", 0)
                simulated_scores[tB["Name"]] = tB.get("FTPS", 0) + settB["bonus"]
                simulated_outcomes[tA["Name"]] = "LOSS"
                simulated_outcomes[tB["Name"]] = "WIN"
                
        else:
            # Solo teams of meer dan 2 teams per GameID (foutje in excel?), doe individuele check
            for t in opponents:
                sett = tier_settings.get(t["OutcomeTier"], {"prob": 50, "bonus": 0})
                # Gebruik prob als ruwe % (bv 90 = 90%)
                is_win = random.random() < (sett["prob"] / 100.0)
                
                score = t.get("FTPS", 0) + (sett["bonus"] if is_win else 0)
                simulated_scores[t["Name"]] = score
                simulated_outcomes[t["Name"]] = "WIN" if is_win else "LOSS"

    return simulated_scores, simulated_outcomes

# ==========================================
# 4. OPTIMIZATION LOOP
# ==========================================

if st.button("🚀 Start Optimalisatie"):
    
    progress_bar = st.progress(0)
    results = []
    prev_lineups = [] 
    
    status_text = st.empty()

    for i in range(num_lineups):
        status_text.text(f"Lineup {i+1} aan het berekenen...")
        
        # --- STAP A: Simuleer de wereld voor deze lineup ---
        sim_scores, sim_outcomes = run_simulation_for_all_games(nba_teams)

        # --- STAP B: Stel het probleem op ---
        prob = LpProblem(f"NBA_Lineup_{i}", LpMaximize)
        x = LpVariable.dicts("Select", [t["Name"] for t in nba_teams], cat="Binary")

        # Objective: Maximize SIMULATED points
        prob += lpSum([x[t["Name"]] * sim_scores[t["Name"]] for t in nba_teams])

        # Constraints
        prob += lpSum([x[t["Name"]] for t in nba_teams]) == team_size
        prob += lpSum([x[t["Name"]] * t["Value"] for t in nba_teams]) <= budget
        
        # Min Diff Constraint (Unieke lineups)
        for prev_set in prev_lineups:
            prob += lpSum([x[name] for name in prev_set]) <= (team_size - min_diff)
            
        # Bracket Constraint: Max 1 per GameID
        if avoid_opposing and "GameID" in df.columns:
            game_ids = set(t["GameID"] for t in nba_teams if pd.notna(t.get("GameID")))
            for gid in game_ids:
                teams_in_game = [t["Name"] for t in nba_teams if t.get("GameID") == gid]
                if len(teams_in_game) > 1:
                    prob += lpSum([x[name] for name in teams_in_game]) <= 1

        prob.solve()

        # --- STAP C: Sla resultaten op ---
        if prob.status == 1:
            selected_names = [t["Name"] for t in nba_teams if x[t["Name"]].value() == 1]
            prev_lineups.append(set(selected_names))
            
            lineup_data = []
            for t in nba_teams:
                if t["Name"] in selected_names:
                    row = t.copy()
                    row["Simulated Points"] = sim_scores[t["Name"]]
                    row["Outcome"] = sim_outcomes[t["Name"]]
                    row["Lineup ID"] = i + 1
                    lineup_data.append(row)
            
            # Sorteren
            lineup_data.sort(key=lambda x: x["Simulated Points"], reverse=True)
            results.extend(lineup_data)
        
        progress_bar.progress((i + 1) / num_lineups)

    status_text.success("Klaar!")
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Toon samenvatting
        cols_to_show = ["Lineup ID", "Name", "GameID", "OutcomeTier", "Outcome", "Simulated Points", "Value"]
        final_cols = [c for c in cols_to_show if c in df_results.columns]
        
        st.subheader("📋 Resultaten")
        
        tab1, tab2 = st.tabs(["Per Lineup", "Excel Data"])
        
        with tab1:
            for lid in range(1, num_lineups + 1):
                subset = df_results[df_results["Lineup ID"] == lid]
                tot_p = subset["Simulated Points"].sum()
                st.markdown(f"**Lineup {lid}** | Totaal Punten (Sim): **{tot_p:.1f}**")
                st.dataframe(subset[final_cols], use_container_width=True)
        
        with tab2:
            st.dataframe(df_results)
            
        # Download
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_results.to_excel(writer, index=False)
        buf.seek(0)
        st.download_button("📥 Download Excel", buf, "nba_optimizer_results.xlsx")
    else:
        st.error("Geen oplossingen gevonden. Check je constraints!")
