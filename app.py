import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
from io import BytesIO
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(page_title="NBA Team Optimizer v3", layout="wide")
st.title("🏀 NBA Team Optimizer v3 (Fixed Win/Loss Points)")
st.markdown("""
**Hoe werkt het?**
1. Upload je Excel met Teams.
2. Stel links in de tabel in hoeveel punten een team krijgt bij **Winst** of **Verlies**.
3. De optimizer simuleert de wedstrijden en kiest de beste combinatie.
""")

# ==========================================
# 1. SIDEBAR: INSTELLINGEN & SCENARIO'S
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
st.sidebar.header("🎲 Tiers & Punten")
st.sidebar.info("Vul hieronder in hoeveel kans (Win %) een Tier heeft, en hoeveel vaste punten ze krijgen bij Winst of Verlies.")

# Standaard configuratie (met jouw voorbeeldwaarden)
default_tier_data = {
    "Tier": [1, 2, 3, 4, 5],
    "Label": ["Heavy Favorite", "Favorite", "Toss Up", "Underdog", "Longshot"],
    "Win %": [90, 70, 50, 30, 10],      # Kans
    "Pts WIN": [450, 450, 450, 450, 450], # Punten als ze winnen
    "Pts LOSS": [200, 200, 200, 200, 200] # Punten als ze verliezen
}

df_tier_config = pd.DataFrame(default_tier_data)

# Bewerkbare tabel in de sidebar
edited_tiers = st.sidebar.data_editor(
    df_tier_config,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Tier": st.column_config.NumberColumn("Tier ID", format="%d", width="small"),
        "Label": st.column_config.TextColumn("Omschrijving", width="medium"),
        "Win %": st.column_config.NumberColumn("Win %", min_value=1, max_value=100),
        "Pts WIN": st.column_config.NumberColumn("Pts bij Winst", min_value=0, format="%d"),
        "Pts LOSS": st.column_config.NumberColumn("Pts bij Verlies", min_value=0, format="%d")
    },
    key="tier_editor"
)

# Settings opslaan in dictionary voor snelle lookup
tier_settings = {}
for index, row in edited_tiers.iterrows():
    try:
        t_id = int(row["Tier"])
        p_win = float(row["Win %"])
        pts_w = float(row["Pts WIN"])
        pts_l = float(row["Pts LOSS"])
        tier_settings[t_id] = {"prob": p_win, "pts_win": pts_w, "pts_loss": pts_l}
    except:
        continue

# ==========================================
# 2. FILE UPLOAD & VALIDATION
# ==========================================

st.markdown("### 1. Upload Teams Data")
uploaded_file = st.file_uploader("Upload Excel bestand", type=["xlsx"])

if not uploaded_file:
    st.info("Upload een Excel bestand met kolommen: Name, Value, OutcomeTier, GameID")
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
    st.warning("⚠️ Geen 'GameID' kolom gevonden. Wedstrijd-koppeling werkt niet optimaal.")
    df["GameID"] = df.index 

# Data types veilig stellen
df["OutcomeTier"] = pd.to_numeric(df["OutcomeTier"], errors='coerce').fillna(3).astype(int)
nba_teams = df.to_dict("records")

with st.expander("🔍 Bekijk geüploade data"):
    st.dataframe(df)

# ==========================================
# 3. SIMULATION LOGIC (FIXED POINTS)
# ==========================================

def run_simulation_for_all_games(teams_data):
    """
    Simuleert wedstrijden.
    De winnaar krijgt 'Pts WIN', de verliezer krijgt 'Pts LOSS'.
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
            games[f"solo_{t['Name']}"].append(t)

    # 2. Speel de wedstrijden
    for gid, opponents in games.items():
        if len(opponents) == 2:
            # Team A vs Team B
            tA = opponents[0]
            tB = opponents[1]
            
            # Haal settings op voor beide teams op basis van hun Tier
            settA = tier_settings.get(tA["OutcomeTier"], {"prob": 50, "pts_win": 0, "pts_loss": 0})
            settB = tier_settings.get(tB["OutcomeTier"], {"prob": 50, "pts_win": 0, "pts_loss": 0})
            
            # Bereken relatieve kans: A / (A + B)
            total_prob = settA["prob"] + settB["prob"]
            if total_prob == 0: total_prob = 1
            
            prob_A_wins = settA["prob"] / total_prob
            
            # Gooi dobbelsteen
            if random.random() < prob_A_wins:
                # A Wint
                simulated_scores[tA["Name"]] = settA["pts_win"]
                simulated_scores[tB["Name"]] = settB["pts_loss"]
                simulated_outcomes[tA["Name"]] = "WIN"
                simulated_outcomes[tB["Name"]] = "LOSS"
            else:
                # B Wint
                simulated_scores[tA["Name"]] = settA["pts_loss"]
                simulated_scores[tB["Name"]] = settB["pts_win"]
                simulated_outcomes[tA["Name"]] = "LOSS"
                simulated_outcomes[tB["Name"]] = "WIN"
                
        else:
            # Solo teams (geen tegenstander in file), simuleer op basis van ruwe kans
            for t in opponents:
                sett = tier_settings.get(t["OutcomeTier"], {"prob": 50, "pts_win": 0, "pts_loss": 0})
                is_win = random.random() < (sett["prob"] / 100.0)
                
                simulated_scores[t["Name"]] = sett["pts_win"] if is_win else sett["pts_loss"]
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
        
        # A. Simuleer punten (Win/Loss) voor alle teams
        sim_scores, sim_outcomes = run_simulation_for_all_games(nba_teams)

        # B. Setup Solver
        prob = LpProblem(f"NBA_Lineup_{i}", LpMaximize)
        x = LpVariable.dicts("Select", [t["Name"] for t in nba_teams], cat="Binary")

        # Doel: Maximaliseer de GESIMULEERDE punten (uit de tabel)
        prob += lpSum([x[t["Name"]] * sim_scores[t["Name"]] for t in nba_teams])

        # Constraints
        prob += lpSum([x[t["Name"]] for t in nba_teams]) == team_size
        prob += lpSum([x[t["Name"]] * t["Value"] for t in nba_teams]) <= budget
        
        # Min Diff Constraint
        for prev_set in prev_lineups:
            prob += lpSum([x[name] for name in prev_set]) <= (team_size - min_diff)
            
        # Max 1 per GameID
        if avoid_opposing and "GameID" in df.columns:
            game_ids = set(t["GameID"] for t in nba_teams if pd.notna(t.get("GameID")))
            for gid in game_ids:
                teams_in_game = [t["Name"] for t in nba_teams if t.get("GameID") == gid]
                if len(teams_in_game) > 1:
                    prob += lpSum([x[name] for name in teams_in_game]) <= 1

        prob.solve()

        # C. Resultaten opslaan
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
            
            # Sorteren op punten
            lineup_data.sort(key=lambda x: x["Simulated Points"], reverse=True)
            results.extend(lineup_data)
        
        progress_bar.progress((i + 1) / num_lineups)

    status_text.success("Klaar!")
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Kolommen netjes ordenen
        cols_to_show = ["Lineup ID", "Name", "GameID", "OutcomeTier", "Outcome", "Simulated Points", "Value"]
        final_cols = [c for c in cols_to_show if c in df_results.columns]
        
        st.subheader("📋 Resultaten")
        
        tab1, tab2 = st.tabs(["Per Lineup", "Excel Data"])
        
        with tab1:
            for lid in range(1, num_lineups + 1):
                subset = df_results[df_results["Lineup ID"] == lid]
                tot_p = subset["Simulated Points"].sum()
                cost = subset["Value"].sum()
                st.markdown(f"**Lineup {lid}** | Punten: **{tot_p:.0f}** | Kosten: **{cost:.1f}**")
                st.dataframe(subset[final_cols], use_container_width=True)
        
        with tab2:
            st.dataframe(df_results)
            
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_results.to_excel(writer, index=False)
        buf.seek(0)
        st.download_button("📥 Download Excel", buf, "nba_fixed_points_results.xlsx")
    else:
        st.error("Geen oplossingen gevonden. Check je constraints!")
