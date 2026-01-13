import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
from io import BytesIO
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(page_title="NBA Team Optimizer v4", layout="wide")
st.title("🏀 NBA Team Optimizer v4")
st.markdown("""
**Features:**
1. **Fixed Points:** Jij bepaalt hoeveel punten winst/verlies oplevert per Tier.
2. **Game Logic:** Max 1 team per wedstrijd (optioneel).
3. **Control:** Forceer teams om mee te doen (Include) of sluit ze uit (Exclude).
""")

# ==========================================
# 1. SIDEBAR: INSTELLINGEN & TIERS
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

# Standaard configuratie
default_tier_data = {
    "Tier": [1, 2, 3, 4, 5],
    "Label": ["Heavy Favorite", "Favorite", "Toss Up", "Underdog", "Longshot"],
    "Win %": [90, 70, 50, 30, 10],      
    "Pts WIN": [450, 450, 450, 450, 450], 
    "Pts LOSS": [200, 200, 200, 200, 200] 
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

# Settings opslaan in dictionary
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
# 2. FILE UPLOAD & PREPARATION
# ==========================================

st.markdown("### 1. Upload Teams Data")
uploaded_file = st.file_uploader("Upload Excel bestand", type=["xlsx"])

nba_teams = []
must_include = []
must_exclude = []
df = pd.DataFrame()

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Check verplichte kolommen
        required_cols = {"Name", "Value", "OutcomeTier"}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Het bestand mist: {required_cols - set(df.columns)}")
            st.stop()
        
        # GameID fix
        if "GameID" not in df.columns:
            st.warning("⚠️ Geen 'GameID' gevonden. Wedstrijd-koppeling staat uit.")
            df["GameID"] = df.index 

        # Data types
        df["OutcomeTier"] = pd.to_numeric(df["OutcomeTier"], errors='coerce').fillna(3).astype(int)
        nba_teams = df.to_dict("records")
        
        # --- NIEUW: INCLUDE / EXCLUDE SELECTORS ---
        # We doen dit pas als het bestand er is, zodat we de teamnamen kennen
        st.sidebar.markdown("---")
        st.sidebar.header("🔒 Forceer Teams")
        
        all_names = sorted(df["Name"].unique())
        
        must_include = st.sidebar.multiselect(
            "Forceer Include (Moet in team)", 
            options=all_names,
            help="Deze teams worden ALTIJD geselecteerd."
        )
        
        # Filter included uit options voor exclude om verwarring te voorkomen
        remain_for_exclude = [n for n in all_names if n not in must_include]
        
        must_exclude = st.sidebar.multiselect(
            "Forceer Exclude (Mag niet in team)", 
            options=remain_for_exclude,
            help="Deze teams worden NOOIT geselecteerd."
        )
        
        with st.expander("🔍 Bekijk geüploade data"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Fout bij verwerken bestand: {e}")
        st.stop()
else:
    st.info("Upload een Excel bestand om de Include/Exclude opties te zien.")
    st.stop()

# ==========================================
# 3. SIMULATION LOGIC
# ==========================================

def run_simulation_for_all_games(teams_data):
    """
    Simuleert wedstrijden. Winnaar krijgt Pts WIN, Verliezer Pts LOSS.
    """
    simulated_scores = {}
    simulated_outcomes = {}
    
    games = defaultdict(list)
    for t in teams_data:
        gid = t.get("GameID")
        if pd.notna(gid):
            games[gid].append(t)
        else:
            games[f"solo_{t['Name']}"].append(t)

    for gid, opponents in games.items():
        if len(opponents) == 2:
            tA = opponents[0]
            tB = opponents[1]
            
            settA = tier_settings.get(tA["OutcomeTier"], {"prob": 50, "pts_win": 0, "pts_loss": 0})
            settB = tier_settings.get(tB["OutcomeTier"], {"prob": 50, "pts_win": 0, "pts_loss": 0})
            
            total_prob = settA["prob"] + settB["prob"]
            if total_prob == 0: total_prob = 1
            prob_A_wins = settA["prob"] / total_prob
            
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
        
        # A. Simulatie
        sim_scores, sim_outcomes = run_simulation_for_all_games(nba_teams)

        # B. Problem Setup
        prob = LpProblem(f"NBA_Lineup_{i}", LpMaximize)
        x = LpVariable.dicts("Select", [t["Name"] for t in nba_teams], cat="Binary")

        # Objective
        prob += lpSum([x[t["Name"]] * sim_scores[t["Name"]] for t in nba_teams])

        # Basis Constraints
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
        
        # --- NIEUW: INCLUDE / EXCLUDE CONSTRAINTS ---
        for name in must_include:
            if name in x:
                prob += x[name] == 1, f"ForceInclude_{name}_{i}"
        
        for name in must_exclude:
            if name in x:
                prob += x[name] == 0, f"ForceExclude_{name}_{i}"

        # Solve
        prob.solve()

        # C. Resultaten
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
            
            lineup_data.sort(key=lambda x: x["Simulated Points"], reverse=True)
            results.extend(lineup_data)
        
        progress_bar.progress((i + 1) / num_lineups)

    status_text.success("Klaar!")
    
    if results:
        df_results = pd.DataFrame(results)
        
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
        st.download_button("📥 Download Excel", buf, "nba_optimizer_v4.xlsx")
    else:
        st.error("Geen oplossingen gevonden. Heb je onmogelijke 'Includes' of Constraints ingesteld?")
