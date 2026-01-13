import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="NBA Team Optimizer (Tiers)", layout="wide")
st.title("🏀 NBA Team Optimizer")
st.markdown("Optimaliseer selecties van NBA Teams op basis van winstkansen en variabele Tiers.")

# ==========================================
# 1. SIDEBAR: INSTELLINGEN & TIERS
# ==========================================

st.sidebar.header("⚙️ Solver Settings")

# Budget & Constraints
budget = st.sidebar.number_input("Max Budget", value=100.0, step=0.5)
team_size = st.sidebar.number_input("Aantal Teams te kiezen", min_value=1, value=5, step=1)
num_lineups = st.sidebar.number_input("Aantal Lineups genereren", min_value=1, max_value=50, value=5)
min_diff = st.sidebar.number_input("Minimaal verschil tussen lineups (aantal teams)", min_value=0, max_value=team_size, value=1)

st.sidebar.markdown("---")
st.sidebar.header("🎲 Probability Tiers")
st.sidebar.info("Stel hieronder je Tiers in. Pas het winstpercentage (Win %) en de bonuspunten bij winst aan.")

# Standaard configuratie voor de Tiers
default_tier_data = {
    "Tier": [1, 2, 3, 4, 5],
    "Label": ["Heavy Favorite", "Solid Favorite", "Coin Flip", "Underdog", "Longshot"],
    "Win %": [90, 75, 50, 30, 15],      # Kans op winst
    "Win Bonus": [5.0, 5.0, 5.0, 5.0, 5.0]  # Punten erbij als ze winnen
}

df_tier_config = pd.DataFrame(default_tier_data)

# Bewerkbare tabel in de sidebar
edited_tiers = st.sidebar.data_editor(
    df_tier_config,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Tier": st.column_config.NumberColumn("Tier ID", format="%d"),
        "Label": st.column_config.TextColumn("Omschrijving"),
        "Win %": st.column_config.NumberColumn("Winstkans (%)", min_value=0, max_value=100),
        "Win Bonus": st.column_config.NumberColumn("Bonus Pts", min_value=0.0, format="%.1f")
    },
    key="tier_editor"
)

# Convert settings to dictionary for fast lookup
# Structure: { TierID: {'prob': 0.8, 'bonus': 5.0}, ... }
tier_settings = {}
for index, row in edited_tiers.iterrows():
    try:
        t_id = int(row["Tier"])
        p_win = float(row["Win %"]) / 100.0
        bonus = float(row["Win Bonus"])
        tier_settings[t_id] = {"prob": p_win, "bonus": bonus}
    except:
        continue

# ==========================================
# 2. FILE UPLOAD & VALIDATION
# ==========================================

st.markdown("### 1. Upload NBA Data")
uploaded_file = st.file_uploader("Upload Excel bestand (Teams)", type=["xlsx"])

if not uploaded_file:
    st.info("Upload een Excel bestand om te beginnen.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Kon bestand niet lezen: {e}")
    st.stop()

# Check required columns
required_cols = {"Name", "Value", "OutcomeTier"}
if not required_cols.issubset(df.columns):
    st.error(f"❌ Het bestand mist verplichte kolommen: {required_cols - set(df.columns)}")
    st.warning("Zorg dat je bestand minstens de kolommen 'Name', 'Value' en 'OutcomeTier' heeft.")
    st.stop()

# Data cleaning
if "FTPS" not in df.columns:
    df["FTPS"] = 0.0 # Base points (zonder winst bonus)

df["OutcomeTier"] = pd.to_numeric(df["OutcomeTier"], errors='coerce').fillna(3).astype(int)
nba_teams = df.to_dict("records")

# Display editor for quick checks
st.markdown("### 2. Controleer Data")
with st.expander("Toon geüploade teams", expanded=False):
    st.dataframe(df)

# ==========================================
# 3. OPTIMIZATION LOGIC
# ==========================================

def simulate_score(team_row):
    """
    Berekent de score voor 1 team in 1 simulatie.
    Kijkt naar de Tier, gooit een 'dobbelsteen' voor winst/verlies,
    en voegt de bonus toe indien gewonnen.
    """
    t_id = team_row["OutcomeTier"]
    base_points = team_row.get("FTPS", 0.0)
    
    # Haal instellingen op uit de sidebar tabel (default naar 50% kans, 0 bonus)
    settings = tier_settings.get(t_id, {"prob": 0.5, "bonus": 0.0})
    
    # De 'Coin Flip' (Random Number Generation)
    is_win = random.random() < settings["prob"]
    
    final_score = base_points + (settings["bonus"] if is_win else 0.0)
    outcome_label = "WIN" if is_win else "LOSS"
    
    return final_score, outcome_label

if st.button("🚀 Genereer NBA Lineups"):
    
    progress_bar = st.progress(0)
    results = []
    prev_lineups = [] # Om te zorgen dat lineups verschillend zijn
    
    status_text = st.empty()

    for i in range(num_lineups):
        status_text.text(f"Optimaliseren van Lineup {i+1} van {num_lineups}...")
        
        # 1. SIMULATIE STAP
        # Voor DEZE lineup simuleren we eerst de scores voor alle teams
        simulated_values = {}
        simulated_outcomes = {}
        
        for team in nba_teams:
            score, outcome = simulate_score(team)
            simulated_values[team["Name"]] = score
            simulated_outcomes[team["Name"]] = outcome

        # 2. SOLVER STAP (PuLP)
        prob = LpProblem(f"NBA_Lineup_{i}", LpMaximize)
        x = LpVariable.dicts("Select", [t["Name"] for t in nba_teams], cat="Binary")

        # Doel: Maximaliseer de GESIMULEERDE punten
        prob += lpSum([x[t["Name"]] * simulated_values[t["Name"]] for t in nba_teams])

        # Constraint: Team Size
        prob += lpSum([x[t["Name"]] for t in nba_teams]) == team_size

        # Constraint: Budget
        prob += lpSum([x[t["Name"]] * t["Value"] for t in nba_teams]) <= budget
        
        # Constraint: Verschil met vorige lineups (Min Diff)
        # Zorgt ervoor dat lineup 2 niet exact lineup 1 is
        for prev_set in prev_lineups:
            prob += lpSum([x[name] for name in prev_set]) <= (team_size - min_diff)

        prob.solve()

        # 3. RESULTAAT OPSLAAN
        if prob.status == 1:
            selected_names = [t["Name"] for t in nba_teams if x[t["Name"]].value() == 1]
            prev_lineups.append(set(selected_names))
            
            # Bouw de resultaat-rijen
            lineup_data = []
            total_pts = 0
            total_cost = 0
            
            for t in nba_teams:
                if t["Name"] in selected_names:
                    row = t.copy()
                    row["Simulated Points"] = simulated_values[t["Name"]]
                    row["Simulated Outcome"] = simulated_outcomes[t["Name"]]
                    row["Lineup ID"] = i + 1
                    lineup_data.append(row)
                    total_pts += row["Simulated Points"]
                    total_cost += row["Value"]
            
            # Sorteren op punten (hoogste eerst) binnen de lineup
            lineup_data.sort(key=lambda x: x["Simulated Points"], reverse=True)
            results.extend(lineup_data)
        
        progress_bar.progress((i + 1) / num_lineups)

    status_text.text("Klaar!")
    
    if not results:
        st.error("Kon geen lineups genereren. Check je budget of constraints.")
    else:
        # ==========================================
        # 4. OUTPUT DISPLAY
        # ==========================================
        df_results = pd.DataFrame(results)
        
        # Kolom volgorde netjes maken
        cols = ["Lineup ID", "Name", "OutcomeTier", "Simulated Outcome", "Simulated Points", "Value", "FTPS"]
        # Pak alleen kolommen die bestaan
        final_cols = [c for c in cols if c in df_results.columns]
        df_display = df_results[final_cols]

        st.subheader(f"📋 Resultaat: {num_lineups} Geoptimaliseerde Lineups")
        
        # Tabbladen voor weergave
        tab1, tab2 = st.tabs(["Samenvatting per Lineup", "Alle Details"])
        
        with tab1:
            # Maak een mooie pivot / groep weergave
            for lid in range(1, num_lineups + 1):
                subset = df_display[df_display["Lineup ID"] == lid]
                tot_p = subset["Simulated Points"].sum()
                tot_c = subset["Value"].sum()
                
                with st.expander(f"Lineup {lid} (Pts: {tot_p:.1f} | Cost: {tot_c:.1f})"):
                    st.dataframe(subset, use_container_width=True)

        with tab2:
            st.dataframe(df_display, use_container_width=True)

        # ==========================================
        # 5. EXCEL DOWNLOAD
        # ==========================================
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_display.to_excel(writer, index=False, sheet_name="NBA Lineups")
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Lineups als Excel",
            data=buf,
            file_name="nba_lineups_optimized.xlsx",
            mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
        )
