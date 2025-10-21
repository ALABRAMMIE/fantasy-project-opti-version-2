import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatusOptimal
import random, re, math
from io import BytesIO
from collections import defaultdict

st.set_page_config(page_title="Fantasy Optimizer v2", layout="wide")
st.title("Fantasy Team Optimizer — v2 (dynamic)")

# --------------------------
# Utilities
# --------------------------
@st.cache_data(show_spinner=False)
def read_excel(file, sheet_name=None, header=0):
    return pd.read_excel(file, sheet_name=sheet_name, header=header)

def split_positions(pos_val):
    """Support multi-position strings like 'PG/SF' or 'MID;FWD'."""
    if pd.isna(pos_val):
        return []
    s = str(pos_val)
    # accept '/', ';', ',', whitespace as separators
    return [p.strip() for p in re.split(r"[\/;,]| {2,}", s.replace(",", "/")) if p.strip()]

def enforce_numeric(series, default=0.0):
    s = pd.to_numeric(series, errors="coerce").fillna(default)
    return s

def kpi(label, value):
    st.metric(label, value)

# --------------------------
# Sidebar: Sport, Template, Solver mode
# --------------------------
sport_options = [
    "-- Choose a sport --", "Cycling", "Speed Skating", "Formula 1", "Stock Exchange",
    "Tennis", "MotoGP", "Football", "Darts", "Cyclocross", "Golf", "Snooker",
    "Olympics", "Basketball", "Dakar Rally", "Skiing", "Rugby", "Biathlon",
    "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment", "Athletics"
]
sport = st.sidebar.selectbox("Select a sport", sport_options)

st.sidebar.markdown("### Upload Profile Template (optioneel, multi-sheet)")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template", type=["xlsx"], key="template_upload_key_v2"
)
available_formats, format_name, team_size_from_format = [], None, None
if template_file:
    try:
        xl = pd.ExcelFile(template_file)
        available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
        if available_formats:
            format_name = st.sidebar.selectbox("Select Format", available_formats)
            m = re.search(r"\((\d+)\)", str(format_name))
            if m:
                team_size_from_format = int(m.group(1))
    except Exception as e:
        st.sidebar.warning(f"⚠️ Template lezen mislukt: {e}")

solver_mode = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"],
    horizontal=False,
)

# --------------------------
# Sidebar: Kernparameters
# --------------------------
budget = st.sidebar.number_input("Max Budget", value=140.0, step=0.1)
default_team_size = team_size_from_format or 13
team_size = st.sidebar.number_input("Team Size", min_value=1, value=default_team_size, step=1)

num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=200, value=5)
diff_count = st.sidebar.number_input("Min verschil tussen teams (# spelers)", min_value=0, max_value=team_size, value=1)

ftps_rand_pct = st.sidebar.slider("FTPS randomness % (teams 2…N)", 0, 100, 10, 1)
rand_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

global_usage_pct = st.sidebar.slider(
    "Global Max Usage % (fallback per speler)",
    0, 100, 100, 5,
    help="Max fractie teams waarin een speler mag voorkomen (Lock forceert 100%)."
)

use_bracket_constraints = st.sidebar.checkbox("Gebruik unieke Bracket per team (mutual exclusivity)")

# --------------------------
# Upload spelers
# --------------------------
st.sidebar.markdown("### Upload Players")
players_file = st.sidebar.file_uploader("Upload Excel (players)", type=["xlsx"], key="players_v2")
if not players_file:
    st.info("Upload je spelersbestand om verder te gaan.")
    st.stop()

try:
    df_raw = read_excel(players_file)
except Exception as e:
    st.error(f"❌ Lezen mislukt: {e}")
    st.stop()

required_cols = {"Name", "Value"}
if not required_cols.issubset(df_raw.columns):
    st.error("❌ Vereiste kolommen ontbreken: 'Name' en 'Value'.")
    st.stop()

# Optional columns
has_ftps = "FTPS" in df_raw.columns
has_pos  = "Position" in df_raw.columns
has_br   = "Bracket" in df_raw.columns
has_team = "Team" in df_raw.columns  # voor toekomstige stacking/limits

df = df_raw.copy()
df["Value"] = enforce_numeric(df["Value"])
if has_ftps:
    df["FTPS"] = enforce_numeric(df["FTPS"])
else:
    df["FTPS"] = 0.0
df["base_FTPS"] = df["FTPS"]

# UI: bewerkbare tabel met Lock/Exclude en per-speler exposure
st.subheader("📋 Spelers bewerken")
editable_cols = ["Name", "Value"]
for c in ["Position", "FTPS", "Bracket", "Team"]:
    if c in df.columns:
        editable_cols.append(c)
# Extra UI kolommen
df["Lock"] = False
df["Exclude"] = False
df["Max Exposure %"] = 100

edited = st.data_editor(
    df[editable_cols + ["Lock", "Exclude", "Max Exposure %"]],
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic"
)
players = edited.to_dict("records")

# Locks / Excludes
include_players = {p["Name"] for p in players if p.get("Lock")}
exclude_players = {p["Name"] for p in players if p.get("Exclude")}
per_player_cap  = {p["Name"]: max(0, min(100, int(p.get("Max Exposure %", 100)))) for p in players}

# --------------------------
# Dynamische constraints: Brackets en Posities
# --------------------------
brackets = sorted({p.get("Bracket") for p in players if p.get("Bracket")})
if use_bracket_constraints and not brackets:
    st.sidebar.warning("⚠️ Unieke Bracket aan, maar geen 'Bracket' kolom.")

# Posities inventariseren
all_positions = set()
if has_pos:
    for p in players:
        for pos in split_positions(p.get("Position")):
            all_positions.add(pos)
all_positions = sorted(all_positions)

# Per-bracket min/max
bracket_min_count, bracket_max_count = {}, {}
if brackets:
    with st.sidebar.expander("Min/Max per Bracket", expanded=False):
        for b in brackets:
            bracket_min_count[b] = st.number_input(f"Bracket {b} — Min", 0, team_size, 0, 1, key=f"min_{b}")
            bracket_max_count[b] = st.number_input(f"Bracket {b} — Max", 0, team_size, team_size, 1, key=f"max_{b}")

# Per-positie min/max
pos_min_count, pos_max_count = {}, {}
if all_positions:
    with st.sidebar.expander("Min/Max per Positie", expanded=True):
        for pos in all_positions:
            pos_min_count[pos] = st.number_input(f"{pos} — Min", 0, team_size, 0, 1, key=f"pmin_{pos}")
            pos_max_count[pos] = st.number_input(f"{pos} — Max", 0, team_size, team_size, 1, key=f"pmax_{pos}")

# --------------------------
# Target profiel voor 'Closest FTP Match'
# --------------------------
target_values = None
if solver_mode == "Closest FTP Match" and template_file and format_name:
    try:
        prof = read_excel(template_file, sheet_name=format_name, header=None)
        raw = prof.iloc[:, 0].dropna().tolist()
        vals = [float(x) for x in raw if isinstance(x, (int, float)) or str(x).replace(".", "", 1).isdigit()]
        if len(vals) < team_size:
            st.error(f"❌ Template heeft minder dan {team_size} waarden.")
            st.stop()
        target_values = vals[:team_size]
    except Exception as e:
        st.error(f"❌ Template lezen mislukt: {e}")
        st.stop()

# --------------------------
# Constraint helpers
# --------------------------
def add_unique_brackets(prob, x):
    if use_bracket_constraints and has_br:
        for b in brackets:
            members = [x[p["Name"]] for p in players if p.get("Bracket") == b]
            if members:
                prob += lpSum(members) <= 1, f"UniqueBracket_{b}"

def add_bracket_minmax(prob, x):
    if has_br and brackets:
        for b in brackets:
            members = [x[p["Name"]] for p in players if p.get("Bracket") == b]
            mn = bracket_min_count.get(b, 0)
            mx = bracket_max_count.get(b, team_size)
            if mn > 0:
                prob += lpSum(members) >= mn, f"MinBracket_{b}"
            if mx < team_size:
                prob += lpSum(members) <= mx, f"MaxBracket_{b}"

def add_position_minmax(prob, x):
    if not has_pos or not all_positions:
        return
    for pos in all_positions:
        members = [x[p["Name"]] for p in players if pos in split_positions(p.get("Position"))]
        if not members:
            continue
        mn = pos_min_count.get(pos, 0)
        mx = pos_max_count.get(pos, team_size)
        if mn > 0:
            prob += lpSum(members) >= mn, f"MinPos_{pos}"
        if mx < team_size:
            prob += lpSum(members) <= mx, f"MaxPos_{pos}"

def add_min_diff(prob, x, team_size, diff_count, prev_sets):
    for idx, prev in enumerate(prev_sets):
        prob += lpSum(x[n] for n in prev) <= team_size - diff_count, f"MinDiff_{idx}"

def add_includes_excludes(prob, x):
    for n in include_players:
        prob += x[n] == 1
    for n in exclude_players:
        prob += x[n] == 0

def exposure_cap_for(name, prev_sets, global_cap_pct, per_player_pct):
    # INCLUDE => cap = num_teams (100%)
    if name in include_players:
        return None  # no cap
    return per_player_pct if per_player_pct is not None else global_cap_pct

def add_exposure_caps(prob, x, num_teams, global_cap_pct, per_player_caps, prev_sets):
    if num_teams <= 1:
        return
    for p in players:
        nm = p["Name"]
        cap_pct = exposure_cap_for(nm, prev_sets, global_cap_pct, per_player_caps.get(nm))
        if cap_pct is None:
            continue
        cap = math.floor(num_teams * cap_pct / 100)
        used = sum(1 for prev in prev_sets if nm in prev)
        prob += (used + x[nm] <= cap, f"Exposure_{nm}")

def feasibility_precheck():
    # Quick checks: locks conflicting with excludes, counts vs team_size, position mins total, etc.
    if include_players & exclude_players:
        return False, "Een speler staat zowel op Lock als Exclude."
    # Som van minimale posities/brackets mag niet > team_size zijn
    total_pos_min = sum(pos_min_count.values()) if pos_min_count else 0
    if total_pos_min > team_size:
        return False, f"Som van Positie-Min ({total_pos_min}) is groter dan Team Size ({team_size})."
    total_br_min = sum(bracket_min_count.values()) if bracket_min_count else 0
    if total_br_min > team_size:
        return False, f"Som van Bracket-Min ({total_br_min}) is groter dan Team Size ({team_size})."
    return True, None

ok, msg = feasibility_precheck()
if not ok:
    st.error(f"🚫 Niet haalbaar: {msg}")
    st.stop()

# --------------------------
# Optimize
# --------------------------
def optimize_lineups():
    random.seed(rand_seed)
    all_teams = []
    prev_sets = []

    # Precompute values
    values = {p["Name"]: float(p["Value"]) for p in players}
    base_ftps = {p["Name"]: float(p["base_FTPS"]) for p in players}

    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for t in range(num_teams):
            prob = LpProblem(f"budget_{t}", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            prob += lpSum(x[n] * values[n] for n in x), "UseBudget"
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * values[n] for n in x) <= upper

            add_unique_brackets(prob, x)
            add_bracket_minmax(prob, x)
            add_position_minmax(prob, x)
            add_exposure_caps(prob, x, num_teams, global_usage_pct, per_player_cap, prev_sets)
            add_min_diff(prob, x, team_size, diff_count, prev_sets)
            add_includes_excludes(prob, x)

            status = prob.solve()
            if status != LpStatusOptimal:
                raise RuntimeError("Infeasible onder huidige constraints.")

            team = [p for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
            upper = sum(values[p["Name"]] for p in team) - 1e-3

    elif solver_mode == "Maximize FTPS":
        for t in range(num_teams):
            if t == 0:
                ftps_vals = base_ftps.copy()
            else:
                ftps_vals = {
                    nm: base_ftps[nm] * (1 + random.uniform(-ftps_rand_pct/100, ftps_rand_pct/100))
                    for nm in base_ftps
                }
            prob = LpProblem(f"ftps_{t}", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            prob += lpSum(x[n] * ftps_vals[n] for n in x), "MaxFTPS"
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * values[n] for n in x) <= budget

            add_unique_brackets(prob, x)
            add_bracket_minmax(prob, x)
            add_position_minmax(prob, x)
            add_exposure_caps(prob, x, num_teams, global_usage_pct, per_player_cap, prev_sets)
            add_min_diff(prob, x, team_size, diff_count, prev_sets)
            add_includes_excludes(prob, x)

            status = prob.solve()
            if status != LpStatusOptimal:
                raise RuntimeError("Infeasible onder huidige constraints.")

            team = [{**p, "Adjusted FTPS": ftps_vals[p["Name"]]} for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})

    else:  # Closest FTP Match
        cap_fallback = math.floor(num_teams * global_usage_pct / 100)
        for _ in range(num_teams):
            slots, used_brackets, used_names = [None]*team_size, set(), set()

            # Place locks eerst
            for n in include_players:
                p0 = next((p for p in players if p["Name"] == n), None)
                if p0 is None:
                    continue
                diffs = [(i, abs(values[n] - target_values[i])) for i in range(team_size) if slots[i] is None]
                best_i = min(diffs, key=lambda x: x[1])[0]
                slots[best_i] = p0
                used_names.add(n)
                if use_bracket_constraints and has_br and p0.get("Bracket"):
                    used_brackets.add(p0["Bracket"])

            # Greedy fill
            for i in range(team_size):
                if slots[i] is not None:
                    continue
                tgt = target_values[i]
                cands = []
                for p in players:
                    nm = p["Name"]
                    if nm in used_names or nm in exclude_players:
                        continue
                    if use_bracket_constraints and has_br and p.get("Bracket") in used_brackets:
                        continue
                    # exposure (per-speler cap > fallback)
                    per_cap = per_player_cap.get(nm, 100)
                    cap_pct = per_cap if per_cap is not None else global_usage_pct
                    cap = math.floor(num_teams * cap_pct / 100)
                    used = sum(1 for prev in prev_sets if nm in prev)
                    if nm not in include_players and used >= cap:
                        continue
                    cands.append(p)
                if not cands:
                    raise RuntimeError("Geen kandidaten meer voor Closest FTP Match (constraints te strak).")
                pick = min(cands, key=lambda p: abs(values[p["Name"]] - tgt))
                slots[i] = pick
                used_names.add(pick["Name"])
                if use_bracket_constraints and has_br and pick.get("Bracket"):
                    used_brackets.add(pick["Bracket"])

            cost = sum(values[p["Name"]] for p in slots if p)
            if cost > budget:
                raise RuntimeError(f"Budget overschreden {cost:.2f} > {budget:.2f}")

            current = {p["Name"] for p in slots if p}
            if prev_sets and len(current & prev_sets[-1]) > team_size - diff_count:
                raise RuntimeError("Min-difference constraint geschonden.")

            team = [{**p, "Adjusted FTPS": p["base_FTPS"]} for p in slots if p]
            all_teams.append(team)
            prev_sets.append(current)
            if len(all_teams) == num_teams:
                break

    return all_teams

col_run, col_reset = st.columns([1,1])
with col_run:
    run = st.button("🚀 Optimize Teams", type="primary")
with col_reset:
    if st.button("🔄 Re-Optimize (zelfde settings)"):
        run = True  # trigger zelfde run

if not run:
    st.stop()

try:
    teams = optimize_lineups()
except Exception as e:
    st.error(f"🚫 Infeasible: {e}")
    st.stop()

# --------------------------
# Output
# --------------------------
st.subheader("📦 Resultaten")
summary_rows = []
for i, team in enumerate(teams, start=1):
    df_t = pd.DataFrame(team)
    # selectie% over alle teams
    df_t["Selectie (%)"] = df_t["Name"].apply(
        lambda n: round(sum(1 for t in teams if any(p["Name"] == n for p in t)) / len(teams) * 100, 1)
    )
    # KPIs
    total_value = df_t["Value"].sum()
    total_ftps  = (df_t["Adjusted FTPS"] if "Adjusted FTPS" in df_t else df_t["base_FTPS"]).sum()
    with st.expander(f"Team {i}", expanded=(i==1)):
        k1, k2, k3 = st.columns(3)
        k1.metric("Budget gebruikt", f"{total_value:.2f} / {budget:.2f}")
        k2.metric("FTPS (som)", f"{total_ftps:.2f}")
        k3.metric("Aantal spelers", len(df_t))
        if has_pos:
            # show count per detected position
            counts = defaultdict(int)
            for _, r in df_t.iterrows():
                for pos in split_positions(r.get("Position")):
                    counts[pos] += 1
            st.caption("Posities in dit team: " + ", ".join([f"{p}:{c}" for p,c in counts.items()]) if counts else "—")
        st.dataframe(df_t, use_container_width=True)

    # voor export + summary
    df_t2 = df_t.copy()
    df_t2["Team"] = i
    summary_rows.append(df_t2)

merged_df = pd.concat(summary_rows, ignore_index=True)

buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    merged_df.to_excel(writer, index=False, sheet_name="All Teams")
    # optioneel: per team als aparte sheets
    for i in range(1, len(teams)+1):
        merged_df[merged_df["Team"] == i].to_excel(writer, index=False, sheet_name=f"Team {i}")
buf.seek(0)

st.download_button(
    "📥 Download (Excel: All + per team)",
    buf,
    file_name="teams_export.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
