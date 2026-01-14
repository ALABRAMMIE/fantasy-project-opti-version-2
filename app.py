import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
from io import BytesIO
from collections import defaultdict

st.set_page_config(page_title="NBA Game Sim Optimizer v7", layout="wide")
st.title("🏀 NBA Game Sim Optimizer v7")
st.markdown("""
**Werking van deze versie:**
1. Het systeem kijkt naar **GameID** om tegenstanders te koppelen.
2. Per lineup wordt de wedstrijd 'gespeeld' op basis van de **Win %** in de tabel.
3. Als Team A wint, krijgt Team B automatisch de verliespunten (en andersom).
4. **ROI Kolom:** Zie direct of een team gekozen is vanwege de punten of de lage prijs.
""")

# ==========================================
# 1. INSTELLINGEN (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Instellingen")

budget = st.sidebar.number_input("Max Budget", value=100.0, step=0.5)
team_size = st.sidebar.number_input("Team Grootte", min_value=1, value=5)
num_lineups = st.sidebar.number_input("Aantal Lineups", min_value=1, max_value=50, value=10)
min_diff = st.sidebar.number_input("Minimaal verschil (spelers)", value=1)
avoid_opposing = st.sidebar.checkbox("Max 1 team per wedstrijd kiezen", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🎲 Kansen & Punten")

# De tabel waarmee jij de logica bepaalt
# Ik heb de verliespunten laag gezet om te voorkomen dat goedkope verliezers gekozen worden
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
    key="tiers_v7"
)

# Settings inladen
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
# 2. UPLOAD & LOGICA
# ==========================================
uploaded_file = st.file_uploader("Upload Excel (Name, Value, OutcomeTier, GameID)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Validatie
    if not {"Name", "Value", "OutcomeTier"}.issubset(df.columns):
        st.error("❌ Mist kolommen: Name, Value of OutcomeTier")
        st.stop()
    
    if "GameID" not in df.columns:
        st.warning("⚠️ Geen GameID gevonden! Simulatie is nu volledig willekeurig (niet gekoppeld).")
        df["GameID"] = df.index
    
    # Zorg dat data types kloppen
    df["OutcomeTier"] = pd.to_numeric(df["OutcomeTier"], errors='coerce').fillna(3).astype(int)
    nba_teams = df.to_dict("records")
    
    # Forceer opties
    all_names = sorted(df["Name"].unique())
    with st.expander("Opties voor Include/Exclude"):
        must_include = st.multiselect("Moet in team:", all_names)
        must_exclude = st.multiselect("Mag niet in team:", [n for n in all_names if n not in must_include])

    # ==========================================
    # 3. DE SIMULATIE ENGINE (Het hart van de code)
    # ==========================================
    def run_matchups(teams_data):
        """
        Speelt alle wedstrijden 1 keer uit.
        Geeft terug: {TeamNaam: Punten}, {TeamNaam: 'WIN'/'LOSS'}
        """
        sim_scores = {}
        sim_outcomes = {}
        
        # 1. Teams groeperen per wedstrijd
        games = defaultdict(list)
        for t in teams_data:
            gid = t.get("GameID")
            if pd.notna(gid): games[gid].append(t)
            else: games[f"solo_{t['Name']}"].append(t) # Fallback

        # 2. Wedstrijden spelen
        for gid, opponents in games.items():
            # Scenario A: Normale wedstrijd (2 teams)
            if len(opponents) == 2:
                tA = opponents[0]
                tB = opponents[1]
                
                # Haal de kracht op uit de tabel
                settA = tier_settings.get(tA["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                settB = tier_settings.get(tB["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                
                # Bereken relatieve winstkans
                # Voorbeeld: A(90) vs B(10) -> Totaal 100. A wint 90/100 keer.
                # Voorbeeld: A(90) vs A(90) -> Totaal 180. A wint 90/180 (50%) keer.
                weight_A = settA["prob"]
                weight_B = settB["prob"]
                total_weight = weight_A + weight_B
                
                if total_weight == 0: total_weight = 1
                prob_A_wins = weight_A / total_weight
                
                # DE DOBBELSTEEN WORP
                if random.random() < prob_A_wins:
                    # A Wint
                    sim_scores[tA["Name"]] = settA["pts_win"]
                    sim_outcomes[tA["Name"]] = "WIN"
                    
                    sim_scores[tB["Name"]] = settB["pts_loss"]
                    sim_outcomes[tB["Name"]] = "LOSS"
                else:
                    # B Wint (Upset!)
                    sim_scores[tA["Name"]] = settA["pts_loss"]
                    sim_outcomes[tA["Name"]] = "LOSS"
                    
                    sim_scores[tB["Name"]] = settB["pts_win"]
                    sim_outcomes[tB["Name"]] = "WIN"

            # Scenario B: Foutje in data of solo team
            else:
                for t in opponents:
                    sett = tier_settings.get(t["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                    # Gewoon ruwe kans gebruiken
                    is_win = random.random() < (sett["prob"] / 100.0)
                    sim_scores[t["Name"]] = sett["pts_win"] if is_win else sett["pts_loss"]
                    sim_outcomes[t["Name"]] = "WIN" if is_win else "LOSS"
        
        return sim_scores, sim_outcomes

    # ==========================================
    # 4. START OPTIMALISATIE
    # ==========================================
    if st.button("🚀 Start Simulatie & Optimalisatie"):
        
        progress = st.progress(0)
        results = []
        prev_lineups = [] # Om unieke lineups te forceren
        
        # Budget check includes
        inc_cost = sum(t["Value"] for t in nba_teams if t["Name"] in must_include)
        if inc_cost > budget:
            st.error(f"Includes te duur ({inc_cost} > {budget})!")
            st.stop()

        for i in range(num_lineups):
            # STAP 1: Speel de wedstrijden voor DEZE lineup
            # Hier gebeurt de magie: elke keer een nieuwe uitslag.
            scores, outcomes = run_matchups(nba_teams)
            
            # STAP 2: Zoek het beste team voor DEZE uitslag
            prob = LpProblem(f"Lineup_{i}", LpMaximize)
            x = LpVariable.dicts("Select", [t["Name"] for t in nba_teams], cat="Binary")
            
            # Objective: Maximizeer punten van deze simulatie
            prob += lpSum([x[t["Name"]] * scores[t["Name"]] for t in nba_teams])
            
            # Constraints
            prob += lpSum([x[t["Name"]] for t in nba_teams]) == team_size
            prob += lpSum([x[t["Name"]] * t["Value"] for t in nba_teams]) <= budget
            
            # Unieke lineups
            for prev in prev_lineups:
                prob += lpSum([x[n] for n in prev]) <= (team_size - min_diff)
            
            # Max 1 per game
            if avoid_opposing:
                # We moeten weten welke teams in welke game zitten
                game_map = defaultdict(list)
                for t in nba_teams:
                    if pd.notna(t.get("GameID")): game_map[t["GameID"]].append(t["Name"])
                for gid, names in game_map.items():
                    if len(names) > 1:
                        prob += lpSum([x[n] for n in names]) <= 1
            
            # Includes / Excludes
            for n in must_include: prob += x[n] == 1
            for n in must_exclude: prob += x[n] == 0
            
            prob.solve()
            
            # STAP 3: Opslaan
            if prob.status == 1:
                selected = [t["Name"] for t in nba_teams if x[t["Name"]].value() == 1]
                prev_lineups.append(set(selected))
                
                for t in nba_teams:
                    if t["Name"] in selected:
                        row = t.copy()
                        row["Simulated Points"] = scores[t["Name"]]
                        row["Simulated Outcome"] = outcomes[t["Name"]]
                        row["Lineup ID"] = i + 1
                        
                        # ROI berekening (Handig voor analyse!)
                        pts = row["Simulated Points"]
                        val = row["Value"]
                        row["ROI"] = round(pts / val, 1) if val > 0 else 0
                        
                        results.append(row)
            
            progress.progress((i+1)/num_lineups)
            
        # ==========================================
        # 5. RESULTATEN
        # ==========================================
        if results:
            df_res = pd.DataFrame(results)
            
            st.success("Klaar! Hieronder de resultaten.")
            
            # Samenvatting per Lineup
            tabs = st.tabs(["Per Lineup", "Excel Data"])
            
            with tabs[0]:
                for lid in range(1, num_lineups + 1):
                    subset = df_res[df_res["Lineup ID"] == lid]
                    t_pts = subset["Simulated Points"].sum()
                    t_cost = subset["Value"].sum()
                    
                    with st.expander(f"Lineup {lid} - Punten: {t_pts} - Kosten: {t_cost}", expanded=(lid==1)):
                        # Laat duidelijk zien wie won en wie verloor in deze simulatie
                        cols = ["Name", "OutcomeTier", "Simulated Outcome", "Simulated Points", "Value", "ROI", "GameID"]
                        st.dataframe(subset[[c for c in cols if c in subset.columns]])

            with tabs[1]:
                st.dataframe(df_res)
                
            # Excel Download
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_res.to_excel(writer, index=False)
            buf.seek(0)
            st.download_button("📥 Download Excel", buf, "nba_results_v7.xlsx")
            
        else:
            st.error("Geen oplossingen gevonden. Check je constraints (budget/includes).")
