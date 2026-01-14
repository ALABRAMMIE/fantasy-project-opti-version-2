import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
from io import BytesIO
from collections import defaultdict

# --- Page Config ---
st.set_page_config(page_title="NBA Game Sim Optimizer v7.1", layout="wide")
st.title("🏀 NBA Game Sim Optimizer v7.1")
st.markdown("""
**Correctie:** De preview van je geüploade Excel-bestand is terug toegevoegd.
De simulatie speelt nog steeds elke wedstrijd 'live' uit per lineup.
""")

# ==========================================
# 1. SIDEBAR INSTELLINGEN
# ==========================================
st.sidebar.header("⚙️ Instellingen")

budget = st.sidebar.number_input("Max Budget", value=100.0, step=0.5)
team_size = st.sidebar.number_input("Team Grootte", min_value=1, value=5)
num_lineups = st.sidebar.number_input("Aantal Lineups", min_value=1, max_value=1000, value=10)
min_diff = st.sidebar.number_input("Minimaal verschil (spelers)", value=1)
avoid_opposing = st.sidebar.checkbox("Max 1 team per wedstrijd kiezen", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🎲 Kansen & Punten")

# De Tiers Tabel
default_data = {
    "Tier": [1, 2, 3, 4, 5],
    "Label": ["Heavy Fav", "Favorite", "Toss Up", "Underdog", "Longshot"],
    "Win %": [90, 70, 50, 30, 10],      
    "Pts WIN": [450, 450, 450, 450, 450], 
    "Pts LOSS": [200, 180, 150, 100, 50] 
}

edited_tiers = st.sidebar.data_editor(
    pd.DataFrame(default_data),
    hide_index=True,
    num_rows="dynamic",
    key="tiers_v7_1"
)

# Settings laden
tier_settings = {}
for i, row in edited_tiers.iterrows():
    try:
        tier_settings[int(row["Tier"])] = {
            "prob": row["Win %"],
            "pts_win": row["Pts WIN"],
            "pts_loss": row["Pts LOSS"]
        }
    except: pass

# ==========================================
# 2. UPLOAD & DATA PREVIEW
# ==========================================
uploaded_file = st.file_uploader("Upload Excel (Name, Value, OutcomeTier, GameID)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Validatie
    if not {"Name", "Value", "OutcomeTier"}.issubset(df.columns):
        st.error("❌ Mist kolommen: Name, Value of OutcomeTier")
        st.stop()
    
    if "GameID" not in df.columns:
        st.warning("⚠️ Geen GameID gevonden! Simulatie is willekeurig.")
        df["GameID"] = df.index
    
    # Types corrigeren
    df["OutcomeTier"] = pd.to_numeric(df["OutcomeTier"], errors='coerce').fillna(3).astype(int)
    nba_teams = df.to_dict("records")
    
    # --- HIER IS DE PREVIEW TERUG ---
    with st.expander("🔍 Bekijk geüploade data", expanded=True):
        st.dataframe(df, use_container_width=True)
    # --------------------------------
    
    # Forceer opties
    all_names = sorted(df["Name"].unique())
    st.sidebar.markdown("---")
    st.sidebar.header("🔒 Forceer Teams")
    must_include = st.sidebar.multiselect("Moet in team (Include):", all_names)
    must_exclude = st.sidebar.multiselect("Mag niet in team (Exclude):", [n for n in all_names if n not in must_include])

    # ==========================================
    # 3. SIMULATIE LOGICA
    # ==========================================
    def run_matchups(teams_data):
        sim_scores = {}
        sim_outcomes = {}
        
        games = defaultdict(list)
        for t in teams_data:
            gid = t.get("GameID")
            if pd.notna(gid): games[gid].append(t)
            else: games[f"solo_{t['Name']}"].append(t)

        for gid, opponents in games.items():
            if len(opponents) == 2:
                tA = opponents[0]
                tB = opponents[1]
                
                settA = tier_settings.get(tA["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                settB = tier_settings.get(tB["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                
                weight_A = settA["prob"]
                weight_B = settB["prob"]
                total_weight = weight_A + weight_B
                if total_weight == 0: total_weight = 1
                prob_A_wins = weight_A / total_weight
                
                if random.random() < prob_A_wins:
                    # A Wint
                    sim_scores[tA["Name"]] = settA["pts_win"]
                    sim_outcomes[tA["Name"]] = "WIN"
                    sim_scores[tB["Name"]] = settB["pts_loss"]
                    sim_outcomes[tB["Name"]] = "LOSS"
                else:
                    # B Wint
                    sim_scores[tA["Name"]] = settA["pts_loss"]
                    sim_outcomes[tA["Name"]] = "LOSS"
                    sim_scores[tB["Name"]] = settB["pts_win"]
                    sim_outcomes[tB["Name"]] = "WIN"
            else:
                for t in opponents:
                    sett = tier_settings.get(t["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                    is_win = random.random() < (sett["prob"] / 100.0)
                    sim_scores[t["Name"]] = sett["pts_win"] if is_win else sett["pts_loss"]
                    sim_outcomes[t["Name"]] = "WIN" if is_win else "LOSS"
        
        return sim_scores, sim_outcomes

    # ==========================================
    # 4. OPTIMIZATION LOOP
    # ==========================================
    if st.button("🚀 Start Simulatie"):
        
        progress = st.progress(0)
        results = []
        prev_lineups = []
        
        # Budget Check
        inc_cost = sum(t["Value"] for t in nba_teams if t["Name"] in must_include)
        if inc_cost > budget:
            st.error(f"Includes te duur ({inc_cost} > {budget})!")
            st.stop()

        for i in range(num_lineups):
            scores, outcomes = run_matchups(nba_teams)
            
            prob = LpProblem(f"Lineup_{i}", LpMaximize)
            x = LpVariable.dicts("Select", [t["Name"] for t in nba_teams], cat="Binary")
            
            prob += lpSum([x[t["Name"]] * scores[t["Name"]] for t in nba_teams])
            prob += lpSum([x[t["Name"]] for t in nba_teams]) == team_size
            prob += lpSum([x[t["Name"]] * t["Value"] for t in nba_teams]) <= budget
            
            for prev in prev_lineups:
                prob += lpSum([x[n] for n in prev]) <= (team_size - min_diff)
            
            if avoid_opposing:
                game_map = defaultdict(list)
                for t in nba_teams:
                    if pd.notna(t.get("GameID")): game_map[t["GameID"]].append(t["Name"])
                for gid, names in game_map.items():
                    if len(names) > 1:
                        prob += lpSum([x[n] for n in names]) <= 1
            
            for n in must_include: prob += x[n] == 1
            for n in must_exclude: prob += x[n] == 0
            
            prob.solve()
            
            if prob.status == 1:
                selected = [t["Name"] for t in nba_teams if x[t["Name"]].value() == 1]
                prev_lineups.append(set(selected))
                
                for t in nba_teams:
                    if t["Name"] in selected:
                        row = t.copy()
                        row["Simulated Points"] = scores[t["Name"]]
                        row["Simulated Outcome"] = outcomes[t["Name"]]
                        row["Lineup ID"] = i + 1
                        val = row["Value"]
                        row["ROI"] = round(row["Simulated Points"] / val, 1) if val > 0 else 0
                        results.append(row)
            
            progress.progress((i+1)/num_lineups)
            
        if results:
            df_res = pd.DataFrame(results)
            st.success("Klaar!")
            
            tabs = st.tabs(["Per Lineup", "Excel Data"])
            with tabs[0]:
                for lid in range(1, num_lineups + 1):
                    subset = df_res[df_res["Lineup ID"] == lid]
                    t_pts = subset["Simulated Points"].sum()
                    t_cost = subset["Value"].sum()
                    with st.expander(f"Lineup {lid} (Pts: {t_pts} | Cost: {t_cost})", expanded=(lid==1)):
                        cols = ["Name", "OutcomeTier", "Simulated Outcome", "Simulated Points", "Value", "ROI", "GameID"]
                        st.dataframe(subset[[c for c in cols if c in subset.columns]])

            with tabs[1]:
                st.dataframe(df_res)
                
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_res.to_excel(writer, index=False)
            buf.seek(0)
            st.download_button("📥 Download Excel", buf, "nba_results_v7_1.xlsx")
        else:
            st.error("Geen oplossingen gevonden.")
