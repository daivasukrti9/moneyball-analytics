"""Moneyball · Baseball Operations & Analytics (2002 Oakland A's Terminal Recreation)."""
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import importlib
from src.app import theme as T
importlib.reload(T)
from src.app import scouting as S
importlib.reload(S)
from src.data.etl import PROCESSED_DIR, build_database_and_cache
from src.models.regression import MoneyballRegressionModel

st.set_page_config(page_title="Moneyball · Baseball Operations", page_icon="⚾", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(T.CSS, unsafe_allow_html=True)

HIT_POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
TARGET_RUNS, TARGET_WINS = 814, 95          # Los números mágicos de DePodesta para 2002
PLOTLY_CFG = {"displayModeBar": False}


# ======================================================================= datos
@st.cache_data(show_spinner=False)
def load_data():
    pp = os.path.join(PROCESSED_DIR, "players_processed.parquet")
    tp = os.path.join(PROCESSED_DIR, "teams_processed.parquet")
    if os.path.exists(pp) and os.path.exists(tp):
        players, teams = pd.read_parquet(pp), pd.read_parquet(tp)
        if "PEI" not in players.columns:
            players, teams = build_database_and_cache()
    else:
        players, teams = build_database_and_cache()
    return players, teams


@st.cache_resource(show_spinner=False)
def get_model(teams: pd.DataFrame) -> MoneyballRegressionModel:
    return MoneyballRegressionModel().fit(teams)


with st.spinner("Loading Lahman Baseball Database..."):
    players, teams = load_data()
    model = get_model(teams)

SEASONS = [int(y) for y in sorted(teams.loc[teams["Payroll"] > 0, "yearID"].unique(), reverse=True)]

# ======================================================================= estado
st.session_state.setdefault("season", 2002)
st.session_state.setdefault("shortlist", [])     # [[playerID, yearID], ...]
st.session_state.setdefault("pool_ver", 0)


@contextmanager
def panel(title: str, key: str, sub: str | None = None):
    with st.container(key=f"panel_{key}"):
        st.markdown(T.panel_header(title, sub), unsafe_allow_html=True)
        yield


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def on_team(df: pd.DataFrame, team_id: str) -> pd.Series:
    return df["teamID"].str.contains(rf"(?:^|/){team_id}(?:/|$)", regex=True)


def lineup_stats(lu: pd.DataFrame) -> tuple[float, float]:
    tb = lu["1B"] + 2 * lu["2B"] + 3 * lu["3B"] + 4 * lu["HR"]
    den = (lu["AB"] + lu["BB"] + lu["HBP"] + lu["SF"]).sum()
    obp = (lu["H"] + lu["BB"] + lu["HBP"]).sum() / den if den else 0.0
    slg = tb.sum() / lu["AB"].sum() if lu["AB"].sum() else 0.0
    return obp, slg


def log_fit(x: np.ndarray, y: np.ndarray):
    b, a = np.polyfit(np.log10(x), y, 1)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 60)
    return xs, a + b * np.log10(xs)


def logo_scatter(df, xcol, ycol, highlight, height, xtitle, ytitle, xfmt, yfmt, hover):
    fig = go.Figure()
    x, y = df[xcol].to_numpy(float), df[ycol].to_numpy(float)
    if len(df) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", hoverinfo="skip",
                                 line=dict(color="#5d655f", width=1.5, dash="dot")))
    pad_x, pad_y = (x.max() - x.min()) * 0.07, (y.max() - y.min()) * 0.10
    hl = df[df["franchID"] == highlight]
    if len(hl):
        fig.add_trace(go.Scatter(x=hl[xcol], y=hl[ycol], mode="markers", hoverinfo="skip",
                                 marker=dict(size=46, color="rgba(243,217,122,.55)",
                                             line=dict(color=T.GOLD_D, width=3))))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=30, opacity=0),
                             customdata=df[["name"]].to_numpy(), hovertemplate=hover + "<extra></extra>"))
    sx, sy = (x.max() - x.min() + 2 * pad_x) * 0.065, (y.max() - y.min() + 2 * pad_y) * 0.10
    for _, r in df.iterrows():
        fig.add_layout_image(dict(source=T.logo_url(r["franchID"]), xref="x", yref="y", x=r[xcol], y=r[ycol],
                                  sizex=sx, sizey=sy, xanchor="center", yanchor="middle", layer="above"))
    T.style_fig(fig, height)
    fig.update_xaxes(title=xtitle, range=[x.min() - pad_x, x.max() + pad_x], tickformat=xfmt)
    fig.update_yaxes(title=ytitle, range=[y.min() - pad_y, y.max() + pad_y], tickformat=yfmt)
    return fig


# ======================================================================= header
with st.container(key="hdr"):
    c_logo, c_title, c_ctrl, c_date = st.columns([0.85, 4.2, 3.7, 2.2], vertical_alignment="center")
    with c_ctrl:
        html('<div class="hdr-tag">FINDING VALUE IN THE NUMBERS</div>')
        s1, s2 = st.columns([1, 2])
        year = s1.selectbox("SEASON", SEASONS, key="season")
        yr_teams = teams[teams["yearID"] == year].sort_values("name")
        franch_opts = yr_teams["franchID"].tolist()
        if st.session_state.get("franch") not in franch_opts:
            st.session_state["franch"] = "OAK" if "OAK" in franch_opts else franch_opts[0]
        names = dict(zip(yr_teams["franchID"], yr_teams["name"]))
        franch = s2.selectbox("CLUB", franch_opts, key="franch", format_func=lambda f: names[f])

    team = yr_teams[yr_teams["franchID"] == franch].iloc[0]
    team_id = team["teamID"]
    with c_logo:
        html(f'<img class="tlogo" src="{T.logo_url(franch)}"/>')
    with c_title:
        html(f'<div class="hdr-title">{team["name"].upper()}</div>'
             f'<div class="hdr-sub">BASEBALL OPERATIONS &amp; ANALYTICS</div>')
    with c_date:
        now = datetime.now()
        date_str = f"{now:%b %d}".upper() + f", {year}"
        html(f'<div class="hdr-date">{T.moneyball_wordmark(19)}<br>{date_str}<br>{now:%I:%M %p}</div>')

# Recortes de datos de la temporada seleccionada
yr_hit = players[(players["yearID"] == year) & (players["Pos"] != "P") & (players["PA"] > 0)]
team_hit = yr_hit[on_team(yr_hit, team_id)]
mlb_obp = yr_teams["OBP"].mean()
div_teams = yr_teams[(yr_teams["lgID"] == team["lgID"]) & (yr_teams["divID"] == team["divID"])]

if "lineup" not in st.session_state:
    st.session_state["lineup"] = team_hit.nlargest(9, "PA")["playerID"].tolist()

tabs = st.tabs(["Ref.", "Player Analysis", "Scouting", "Team Trends", "Market Value", "Roster Options", "Reports"])

# ======================================================================= 0 · REF & GLOSSARY
with tabs[0]:
    r = st.columns([5.2, 4.8])
    with r[0]:
        with panel("Glossary & Sabermetric Acronyms", "glos", "core formulas & metrics"):
            glossary_html = """
            <table class="mbt">
                <tr><th>Acronym</th><th>Full Metric Name</th><th>Formula / Definition</th><th class="n">MLB Benchmark</th></tr>
                <tr><td><b>OBP</b></td><td>On-Base Percentage</td><td>(H + BB + HBP) / (AB + BB + HBP + SF)</td><td class="n">.320 avg | .380+ elite</td></tr>
                <tr><td><b>SLG</b></td><td>Slugging Percentage</td><td>Total Bases / At Bats (AB)</td><td class="n">.415 avg | .500+ elite</td></tr>
                <tr><td><b>OPS</b></td><td>On-Base Plus Slugging</td><td>OBP + SLG</td><td class="n">.735 avg | .850+ elite</td></tr>
                <tr><td><b>PEI</b></td><td>Player Efficiency Index</td><td>100 × (Player RC/PA) / (League RC/PA)</td><td class="n">100 = League Average</td></tr>
                <tr><td><b>RC</b></td><td>Runs Created (Bill James)</td><td>(H + BB) × (1B + 2·2B + 3·3B + 4·HR) / (AB + BB)</td><td class="n">75 avg | 100+ elite</td></tr>
                <tr><td><b>RC/600</b></td><td>Runs Created per 600 PA</td><td>(Runs Created / PA) × 600 plate appearances</td><td class="n">Full Season Baseline</td></tr>
                <tr><td><b>PA</b></td><td>Plate Appearances</td><td>AB + BB + HBP + SF + SH</td><td class="n">True Sample Size</td></tr>
                <tr><td><b>AB</b></td><td>At Bats (Official Turn)</td><td>PA excluding walks, HBP and sacrifices</td><td class="n">AVG & SLG denominator</td></tr>
                <tr><td><b>$/RC</b></td><td>Cost per Run Created</td><td>Player Salary / Runs Created (RC)</td><td class="n">&lt;$25K/RC = Value Target</td></tr>
                <tr><td><b>Pyth W%</b></td><td>Pythagorean Expectation</td><td>R² / (R² + RA²)</td><td class="n">Expected Wins (luck-free)</td></tr>
                <tr><td><b>GB</b></td><td>Games Behind</td><td>((W_leader - W) + (L - L_leader)) / 2</td><td class="n">Division deficit</td></tr>
            </table>
            """
            html(glossary_html)

        with panel("System Ground Rules · Billy Beane & Bill James", "rules"):
            html(T.bullets([
                "Rule 1: We do not buy players, we buy runs. The scarcest and most finite asset in baseball is 27 outs per game; surrendering an out is the highest offensive cost.",
                "Rule 2: Recreate the star in the aggregate. A superstar slugger loss (Jason Giambi) isn't replaced with an expensive name, but by summing accessible parts (Hatteberg + Justice + J. Giambi) at 20% of the cost.",
                "Rule 3: Trust sample size convergence (162 games). Bad slumps in April (<100 PA) are stochastic noise; past 460+ PA variance collapses and mathematical probability reigns.",
                "Rule 4: Exploit market price asymmetry. The conventional market overpays for aesthetics, clean swings, and batting average (AVG); the system pays for OBP and plate discipline."
            ]))

    with r[1]:
        with panel("Terminal Guide · Architecture by Tab", "guide", "quick system overview"):
            guide_items = [
                ("1. Player Analysis", "1:1 Movie Terminal overview: Player Efficiency Index (PEI) leaderboard, Runs Created vs. Cost log scatter with star value indicator, Key Insights, Probability Model, On-Base Advantage comparison, Lineup Options, and Roster Strategy."),
                ("2. Scouting", "BATS Technical Scouting System: Player Bio Dossier (Ficha Técnica) with CRT portrait, 3x3 interactive strike zone with pitch outcome markers (H, HR, S, B, F), bat orientation (R/L), pitch breakdown matrix, inset beveled statistics, pitching arsenal, and baserunning diamond."),
                ("3. Team Trends", "Interactive 30-club map with official logos (Payroll vs Wins), division standings with GB, historical franchise payroll trends, and OBP vs Runs correlation."),
                ("4. Market Value", "Sabermetric scouting filters (PA, Salary, OBP), market inefficiency scatter map, and interactive Target Board to build a shopping Shortlist."),
                ("5. Roster Options", "General Manager simulator: checkbox selection for 9 hitters, real-time runs allowed and budget sliders, projecting expected wins via the Pythagorean model."),
                ("6. Reports", "Dynamic Roster Turnover (Departures vs Acquisitions), Executive Season Audit (W-L, Pythagorean luck, payroll efficiency), 20-season Monte Carlo convergence, and front-office CSV exports.")
            ]
            rows_guide = "".join(f'<div class="r" style="display:block; margin-bottom:8px; padding-bottom:6px; border-bottom:1px dotted #b9b6a6;">'
                                 f'<b style="color:#1e3a2f; display:block; font-size:14px;">{k}</b>'
                                 f'<span style="font-size:12.5px; color:#3d4a43; line-height:1.35;">{v}</span></div>'
                                 for k, v in guide_items)
            html(f'<div class="kv">{rows_guide}</div>')

        with panel("Data Pipeline & Methodology", "meth", "data sources & integrity"):
            html(T.kv_rows([
                ("Primary Source", "Lahman Baseball Database (1960–2023)"),
                ("Mirror Repository", "Chadwick Baseball Bureau (BaseballDataBank)"),
                ("Relational Keys", "playerID, yearID, teamID composite keys"),
                ("Pre-1954 Sanitation", "Conditional imputation SF = 0 (unrecorded pre-1954)"),
                ("Mid-Season Trades", "Multi-stint aggregation by player-season (prevents dilution)"),
                ("Zero-Division Guard", "Vectorized safety via np.where on PA=0 and AB=0"),
                ("Predictive Engine", "Calibrated Scikit-Learn OLS (R = β0 + β1·OBP + β2·SLG)"),
            ]))

# ======================================================================= 1 · PLAYER ANALYSIS
with tabs[1]:
    r1 = st.columns([4.3, 3.7, 2.6])
    with r1[0], panel("Player Efficiency Index (PEI)", "pei", "100 = MLB avg"):
        scope = st.segmented_control("scope", ["TEAM", "MLB"], default="TEAM", key="pei_scope",
                                     label_visibility="collapsed") or "TEAM"
        src = team_hit[team_hit["PA"] >= 150] if scope == "TEAM" else yr_hit[yr_hit["PA"] >= 400]
        top10 = src.nlargest(10, "PEI")
        html(T.rank_table(top10, show_team=scope == "MLB"))

    paid = yr_hit[(yr_hit["PA"] >= 300) & (yr_hit["salary"] > 0)]
    team_paid = team_hit[(team_hit["PA"] >= 300) & (team_hit["salary"] > 0)]
    star = team_paid.nlargest(1, "PEI")

    with r1[1], panel("Runs Created vs. Cost", "rcc", "RC per 600 PA"):
        fig = go.Figure()
        if len(paid) > 3:
            xs, ys = log_fit(paid["salary"].to_numpy() / 1e6, paid["RC_600"].to_numpy())
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", line=dict(color="#4b524d", width=1.6)))
        for df_, color, size in ((paid[~paid["playerID"].isin(team_paid["playerID"])], T.GRAY, 7),
                                 (team_paid, T.GREEN, 9)):
            fig.add_trace(go.Scatter(
                x=df_["salary"] / 1e6, y=df_["RC_600"], mode="markers",
                marker=dict(size=size, color=color, opacity=.8, line=dict(color="#3d423e", width=.6)),
                customdata=df_[["fullName", "teamID"]].to_numpy(),
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>$%{x:.2f}M · %{y:.1f} RC/600<extra></extra>"))
        if len(star):
            fig.add_trace(go.Scatter(
                x=star["salary"] / 1e6, y=star["RC_600"], mode="markers+text", text=["  " + star.iloc[0]["nameLast"]],
                textposition="middle right", textfont=dict(size=14, color=T.INK),
                marker=dict(size=15, color=T.GOLD, line=dict(color=T.INK, width=1.5)), hoverinfo="skip"))
        T.style_fig(fig, 352)
        fig.update_xaxes(type="log", title="Salary (Millions USD)", tickvals=[0.3, 0.5, 1, 2, 5, 10, 20],
                         ticktext=["0.3", "0.5", "1.0", "2.0", "5", "10", "20"])
        fig.update_yaxes(title="Runs Created per 600 PA")
        st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CFG)

    with r1[2]:
        with panel("Key Insights", "ins"):
            cpw_rank = int(yr_teams["Cost_per_Win"].rank().loc[team.name])
            items = [
                "Targets: undervalued players with high on-base skills.",
                "Focus on OBP, not just home runs.",
                "Build depth with players who get on base."
            ]
            if len(star):
                s = star.iloc[0]
                items.append(f"Top value: {s['shortName']} · {s['RC_600']:.0f} RC/600 at {T.money(s['salary'])}")
            items.append(f"Team OBP {T.f3(team['OBP'])} vs MLB {T.f3(mlb_obp)} ({(team['OBP'] - mlb_obp) * 1000:+.0f} pts)")
            items.append(f"Cost per win ${team['Cost_per_Win'] / 1e3:,.0f}K · #{cpw_rank} of {len(yr_teams)}")
            html(T.bullets(items))
        with panel("Probability Model", "prob"):
            pyth = team["R"] ** 2 / (team["R"] ** 2 + team["RA"] ** 2)
            div_sorted = div_teams.sort_values(["W", "L"], ascending=[False, True]).reset_index(drop=True)
            leader = div_sorted.iloc[0]
            if leader["franchID"] == franch:
                second = div_sorted.iloc[1] if len(div_sorted) > 1 else leader
                gb_text = f"+{((team['W'] - second['W']) + (second['L'] - team['L'])) / 2:.1f} GB"
            else:
                gb_val = ((leader["W"] - team["W"]) + (team["L"] - leader["L"])) / 2
                gb_text = f"{gb_val:.1f} GB"

            win_prob = f"{pyth * 100:.1f}%"
            playoff_odds = f"{min(99.9, max(5.0, (team['Pythagorean_Wins'] - 75) * 4.2)):.1f}%"

            html(T.kv_rows([
                ("Win Probability (Next Game)", win_prob),
                ("Playoff Odds", playoff_odds),
                ("Division Lead / Standing", gb_text),
                ("Expected Wins", f"{team['Pythagorean_Wins']:.1f}"),
                ("Actual Wins", f"{int(team['W'])}"),
                ("Run Differential", f"{int(team['R'] - team['RA']):+d}"),
                ("Luck Factor", f"{team['W'] - team['Pythagorean_Wins']:+.1f} W"),
            ]))

    r2 = st.columns([4.3, 3.7, 2.6])
    with r2[0], panel("On-Base Advantage", "oba"):
        div_label = f"{team['lgID']} {dict(E='East', W='West', C='Central').get(team['divID'], team['divID'])}"
        vals = [team["OBP"], mlb_obp, div_teams["OBP"].mean()]
        labels = [team_id, "MLB Avg", div_label]
        fig = go.Figure(go.Bar(x=labels, y=vals, text=[T.f3(v) for v in vals], textposition="outside",
                               marker=dict(color=["#1f4a36", "#5f8a72", "#9dbaa6"], line=dict(color="#142a20", width=1)),
                               width=.62, hoverinfo="skip"))
        T.style_fig(fig, 228, margin=dict(l=52, r=12, t=14, b=34))
        fig.update_yaxes(range=[0, max(vals) * 1.18], tickformat=".3f")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CFG)

    with r2[1], panel("Lineup Options", "lo", "Projected Runs / 600 PA"):
        opts = team_hit[team_hit["PA"] >= 200].nlargest(5, "RC_600")
        html(T.value_list([(f"{r['nameLast']} {r['Pos']}", f"{r['RC_600']:.1f}") for _, r in opts.iterrows()]))

    with r2[2], panel("Roster Strategy", "rs"):
        html(T.bullets(["Prioritize OBP and versatility.", "Explore undervalued free agents.",
                        "Optimize lineup for run creation, not home run totals."]))

# ======================================================================= 2 · SCOUTING
with tabs[2]:
    avail_players = team_hit.copy()
    if len(avail_players) == 0:
        avail_players = yr_hit.head(20).copy()

    c_quick = st.columns([1.1, 5.7, 3.2], vertical_alignment="center")
    c_quick[0].markdown("<b style='font-size:12px; color:#1e3a2f; letter-spacing:.08em;'>SCOUT TARGET:</b>", unsafe_allow_html=True)

    default_pid = "giambje01" if "giambje01" in set(avail_players["playerID"]) else avail_players.iloc[0]["playerID"]
    if "scout_pid" not in st.session_state or st.session_state["scout_pid"] not in set(players["playerID"]):
        st.session_state["scout_pid"] = default_pid

    with c_quick[1]:
        quick_pids = [
            ("Jeremy Giambi", "giambje01"),
            ("Scott Hatteberg", "hattesc01"),
            ("David Justice", "justida01"),
            ("Miguel Tejada", "tejadmi01"),
            ("Eric Chavez", "chaveer01"),
            ("Chad Bradford", "bradfch01"),
        ]
        q_cols = st.columns(len(quick_pids))
        for i, (q_name, q_pid) in enumerate(quick_pids):
            btn_type = "primary" if st.session_state["scout_pid"] == q_pid else "secondary"
            if q_cols[i].button(q_name.split()[0], key=f"q_{q_pid}", width="stretch", type=btn_type):
                st.session_state["scout_pid"] = q_pid
                st.rerun()

    with c_quick[2]:
        p_dict = dict(zip(avail_players["playerID"], avail_players["fullName"]))
        if st.session_state["scout_pid"] not in p_dict:
            p_match = players[(players["playerID"] == st.session_state["scout_pid"]) & (players["yearID"] == year)]
            if len(p_match):
                p_dict[st.session_state["scout_pid"]] = p_match.iloc[0]["fullName"]
            else:
                p_match2 = players[players["playerID"] == st.session_state["scout_pid"]]
                p_dict[st.session_state["scout_pid"]] = p_match2.iloc[0]["fullName"] if len(p_match2) else "Jeremy Giambi"

        p_keys = list(p_dict.keys())
        idx = p_keys.index(st.session_state["scout_pid"]) if st.session_state["scout_pid"] in p_keys else 0
        sel_pid = st.selectbox("PLAYER", p_keys, index=idx, format_func=lambda k: p_dict.get(k, k),
                               key="scout_sel_box", label_visibility="collapsed")
        if sel_pid != st.session_state["scout_pid"]:
            st.session_state["scout_pid"] = sel_pid
            st.rerun()

    cur_pid = st.session_state["scout_pid"]
    p_row_match = players[(players["playerID"] == cur_pid) & (players["yearID"] == year)]
    if len(p_row_match):
        p_row = p_row_match.iloc[0]
    else:
        p_row_all = players[players["playerID"] == cur_pid]
        p_row = p_row_all.iloc[-1] if len(p_row_all) else avail_players.iloc[0]

    bio = S.get_player_bio(cur_pid, p_row)

    c_bar = st.columns([4.2, 1.8, 1.8, 1.2, 1.0], vertical_alignment="center")
    scout_scope = c_bar[0].segmented_control(
        "SCOUT MODE",
        ["BATTING & CONTACT ZONE", "PITCHING & ARSENAL", "BASES & SPEED"],
        default="BATTING & CONTACT ZONE",
        key="scout_scope_ctl",
        label_visibility="collapsed"
    )
    p_type_filter = c_bar[1].selectbox("Pitch Type", ["ALL", "FASTBALL", "SLIDER", "CHANGEUP", "CURVEBALL"], key="p_type_flt")
    res_filter = c_bar[2].selectbox("Result", ["ALL", "HITS ONLY (H/HR)", "STRIKES (S)", "BALLS (B)"], key="p_res_flt")
    inn_filter = c_bar[3].selectbox("Inn", ["ALL", "1-3", "4-6", "7-9"], key="p_inn_flt")
    if c_bar[4].button("RESET", key="flt_reset", width="stretch"):
        st.session_state["p_type_flt"] = "ALL"
        st.session_state["p_res_flt"] = "ALL"
        st.session_state["p_inn_flt"] = "ALL"
        st.rerun()

    all_pitches = S.generate_scouting_pitches(p_row)
    filtered_pitches = all_pitches.copy()
    if p_type_filter != "ALL":
        filtered_pitches = filtered_pitches[filtered_pitches["pitch_type"] == p_type_filter]
    if res_filter == "HITS ONLY (H/HR)":
        filtered_pitches = filtered_pitches[filtered_pitches["result"].isin(["H", "HR"])]
    elif res_filter == "STRIKES (S)":
        filtered_pitches = filtered_pitches[filtered_pitches["result"] == "S"]
    elif res_filter == "BALLS (B)":
        filtered_pitches = filtered_pitches[filtered_pitches["result"] == "B"]
    if inn_filter != "ALL":
        filtered_pitches = filtered_pitches[filtered_pitches["inn"] == inn_filter]

    col_s = st.columns([3.5, 3.8, 4.7])

    with col_s[0]:
        with panel("Scouting Dossier · Ficha Técnica", "bio", f"{bio['fullName'].upper()} · {bio['Pos']}"):
            html(S.get_bio_card_html(bio))
            pei_val = float(p_row.get("PEI", 100))
            pei_gold = pei_val >= 115
            pei_sub = "42% above league baseline" if pei_val >= 140 else ("Above average run creator" if pei_val >= 110 else "Disciplined on-base contributor")
            html(f'<div style="margin-top:8px;">'
                 + T.tile("EFFICIENCY INDEX (PEI)", f"{pei_val:.1f}", pei_sub, pei_val / 160.0, gold=pei_gold)
                 + '</div>')

    with col_s[1]:
        if scout_scope == "BASES & SPEED":
            with panel("Base Running & Diamond Analysis", "dia", "stolen base efficiency & baserunning value"):
                sb = int(p_row.get("SB", 3))
                cs = int(p_row.get("CS", 1))
                fig_dia = S.render_diamond_fig(sb, cs)
                st.plotly_chart(fig_dia, theme=None, width="stretch", config=PLOTLY_CFG, key=f"dia_{cur_pid}_{sb}_{cs}")
                html('<div style="font-size:11.5px; color:#3c4a42; text-align:center; font-weight:600; margin-top:2px;">'
                     'Diamond tracking: Steal paths, extra-base frequency &amp; out avoidance'
                     '</div>')
        else:
            title_zone = "Scouting · Pitch Matrix" if scout_scope == "BATTING & CONTACT ZONE" else "Pitcher Arsenal & Release Zone"
            sub_zone = f"{len(filtered_pitches)} pitches plotted · 3x3 strike grid"
            with panel(title_zone, "zone", sub_zone):
                fig_zone = S.render_strike_zone_fig(filtered_pitches, bio["bats"])
                st.plotly_chart(fig_zone, theme=None, width="stretch", config=PLOTLY_CFG, key=f"sz_{cur_pid}_{bio['bats']}_{len(filtered_pitches)}")
                html('<div style="font-size:11.5px; color:#4a544d; text-align:center; font-style:italic; margin-top:4px;">'
                     'Click on pitch dots to inspect velocity, pitch type, and count details.'
                     '</div>')

    with col_s[2]:
        if scout_scope == "BATTING & CONTACT ZONE":
            with panel("Pitch Breakdown & Statistics", "brk", f"season profile · {int(p_row.get('PA', 156))} PA"):
                html(S.get_pitch_breakdown_html(filtered_pitches, int(p_row.get("PA", 156))))
                html(S.get_boxed_statistics_html(p_row))
        elif scout_scope == "PITCHING & ARSENAL":
            with panel("Pitching Arsenal & Leverage Audit", "parm", f"arsenal mechanics · {franch} {year}"):
                html(S.get_pitcher_arsenal_html(cur_pid))
                html(S.get_pitcher_statistics_html(cur_pid, year))
        else:
            with panel("Baserunning Discipline & Steal Matrix", "barm", "run preservation & base advancement"):
                html(S.get_baserunning_statistics_html(p_row))

# ======================================================================= 3 · TEAM TRENDS
with tabs[3]:
    r1 = st.columns([6.4, 3.6])
    with r1[0], panel(f"Payroll vs. Wins · {year}", "pw", "MLB"):
        d = yr_teams.assign(PayM=yr_teams["Payroll"] / 1e6)
        fig = logo_scatter(d, "PayM", "W", franch, 440, "Payroll (Millions USD)", "Wins", ".0f", ".0f",
                           "%{customdata[0]}<br>$%{x:.1f}M · %{y} W")
        st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CFG)

    with r1[1]:
        with panel(f"Standings · {div_label}", "std"):
            dv = div_teams.sort_values(["W", "L"], ascending=[False, True])
            lw, ll = dv.iloc[0]["W"], dv.iloc[0]["L"]
            rows = ""
            for _, t_ in dv.iterrows():
                gb = "—" if t_["W"] == lw and t_["L"] == ll else f"{((lw - t_['W']) + (t_['L'] - ll)) / 2:.1f}"
                hl = ' class="hl"' if t_["franchID"] == franch else ""
                rows += (f'<tr{hl}><td><img src="{T.logo_url(t_["franchID"])}"/>{t_["teamID"]}</td>'
                         f'<td class="n">{int(t_["W"])}</td><td class="n">{int(t_["L"])}</td>'
                         f'<td class="n">{gb}</td><td class="n">{T.money(t_["Payroll"])}</td></tr>')
            html('<table class="mbt"><tr><th>Team</th><th class="n">W</th><th class="n">L</th>'
                 f'<th class="n">GB</th><th class="n">Payroll</th></tr>{rows}</table>')
        with panel("Efficiency", "eff"):
            pay_rank = int(yr_teams["Payroll"].rank(ascending=False).loc[team.name])
            w_rank = int(yr_teams["W"].rank(ascending=False, method="min").loc[team.name])
            html(T.kv_rows([("Payroll Rank", f"#{pay_rank} of {len(yr_teams)}"),
                            ("Wins Rank", f"#{w_rank} of {len(yr_teams)}"),
                            ("Cost per Win", f"${team['Cost_per_Win'] / 1e3:,.0f}K"),
                            ("MLB Avg Cost per Win", f"${yr_teams['Cost_per_Win'].mean() / 1e3:,.0f}K")]))

    r2 = st.columns(2)
    with r2[0], panel(f"{team_id} · Wins & Payroll History", "hist"):
        h = teams[(teams["franchID"] == franch) & (teams["Payroll"] > 0)].sort_values("yearID")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=h["yearID"], y=h["Payroll"] / 1e6, name="Payroll",
                             marker=dict(color=[T.GOLD_D if y == year else "#a9c2b0" for y in h["yearID"]]),
                             hovertemplate="%{x}: $%{y:.1f}M<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=h["yearID"], y=h["W"], mode="lines+markers", name="Wins",
                                 line=dict(color=T.GREEN_D, width=2.2), marker=dict(size=5),
                                 hovertemplate="%{x}: %{y} W<extra></extra>"), secondary_y=True)
        T.style_fig(fig, 300, margin=dict(l=52, r=48, t=10, b=36))
        fig.update_yaxes(title="Payroll ($M)", secondary_y=False)
        fig.update_yaxes(title="Wins", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CFG)

    with r2[1]:
        corr = np.corrcoef(yr_teams["OBP"], yr_teams["R"])[0, 1]
        with panel(f"OBP vs. Runs · {year}", "obpr", f"r = {corr:.2f}"):
            fig = logo_scatter(yr_teams, "OBP", "R", franch, 300, "Team OBP", "Runs Scored", ".3f", ".0f",
                               "%{customdata[0]}<br>OBP %{x:.3f} · %{y} R")
            st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CFG)

# ======================================================================= 4 · MARKET VALUE
with tabs[4]:
    with panel("Scouting Filters", "flt"):
        f = st.columns([1.2, 1.2, 1.2, 2.4])
        min_pa = f[0].slider("Min PA", 100, 650, 300, 25)
        max_sal = f[1].slider("Max Salary ($M)", 0.3, 25.0, 2.0, 0.1)
        min_obp = f[2].slider("Min OBP", 0.280, 0.420, 0.340, 0.005, format="%.3f")
        pos_sel = f[3].multiselect("Positions", HIT_POS + ["UT"], default=HIT_POS + ["UT"])

    mk = yr_hit[(yr_hit["PA"] >= min_pa) & (yr_hit["salary"] > 0)]
    targets = mk[(mk["salary"] <= max_sal * 1e6) & (mk["OBP"] >= min_obp) & (mk["Pos"].isin(pos_sel))] \
        .sort_values("PEI", ascending=False)

    r = st.columns([5.2, 4.8])
    with r[0], panel("Market Inefficiency Map", "map", f"{len(targets)} targets"):
        fig = go.Figure()
        if len(mk) > 3:
            xs, ys = log_fit(mk["salary"].to_numpy() / 1e6, mk["RC_600"].to_numpy())
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", line=dict(color="#4b524d", width=1.6)))
        others = mk[~mk["playerID"].isin(targets["playerID"])]
        for df_, color, size in ((others, "#9a9a90", 7), (targets, T.GOLD, 11)):
            fig.add_trace(go.Scatter(
                x=df_["salary"] / 1e6, y=df_["RC_600"], mode="markers",
                marker=dict(size=size, color=color, line=dict(color="#3d423e", width=.8)),
                customdata=df_[["fullName", "teamID", "OBP"]].to_numpy(),
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>$%{x:.2f}M · %{y:.1f} RC/600 · OBP %{customdata[2]:.3f}<extra></extra>"))
        top3 = targets.head(3)
        fig.add_trace(go.Scatter(x=top3["salary"] / 1e6, y=top3["RC_600"], mode="text", text=top3["nameLast"],
                                 textposition="top center", textfont=dict(size=12.5, color=T.INK), hoverinfo="skip"))
        fig.add_vline(x=np.log10(max_sal), line=dict(color=T.GOLD_D, dash="dash", width=1.5))
        T.style_fig(fig, 430)
        fig.update_xaxes(type="log", title="Salary (Millions USD)", tickvals=[0.3, 0.5, 1, 2, 5, 10, 20],
                         ticktext=["0.3", "0.5", "1.0", "2.0", "5", "10", "20"])
        fig.update_yaxes(title="Runs Created per 600 PA")
        st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CFG)

    with r[1], panel("Target Board", "board", "select rows → shortlist"):
        board = pd.DataFrame({
            "Player": targets["fullName"], "Tm": targets["teamID"], "Pos": targets["Pos"], "PA": targets["PA"].astype(int),
            "OBP": targets["OBP"], "SLG": targets["SLG"], "PEI": targets["PEI"], "RC/600": targets["RC_600"],
            "Salary": targets["salary"] / 1e6, "$/RC": targets["Cost_per_RC"] / 1e3,
        })
        ev = st.dataframe(
            board, key=f"board_{year}", on_select="rerun", selection_mode="multi-row", hide_index=True, height=345,
            column_config={
                "OBP": st.column_config.NumberColumn(format="%.3f"), "SLG": st.column_config.NumberColumn(format="%.3f"),
                "PEI": st.column_config.NumberColumn(format="%.1f"), "RC/600": st.column_config.NumberColumn(format="%.1f"),
                "Salary": st.column_config.NumberColumn("Salary $M", format="%.2f"),
                "$/RC": st.column_config.NumberColumn("$K / RC", format="%.1f"),
            })
        picked_rows = ev.selection.rows
        if st.button(f"＋ ADD TO SHORTLIST ({len(picked_rows)})", disabled=not picked_rows, width="stretch"):
            have = {tuple(x) for x in st.session_state["shortlist"]}
            for pid in targets.iloc[picked_rows]["playerID"]:
                if (pid, year) not in have:
                    st.session_state["shortlist"].append([pid, int(year)])
            st.rerun()

    sl = pd.DataFrame(st.session_state["shortlist"], columns=["playerID", "yearID"])
    sl_df = sl.merge(players, on=["playerID", "yearID"], how="left")
    with panel("Shortlist", "sl", f"{len(sl_df)} players"):
        c = st.columns([8.5, 1.5], vertical_alignment="center")
        with c[0]:
            if len(sl_df):
                html('<div class="chips">' + "".join(
                    f'<span class="chip on">{r["shortName"]} · {r["Pos"]} · {T.f3(r["OBP"])} · {T.money(r["salary"])} · {r["yearID"]}</span>'
                    for _, r in sl_df.iterrows()) + "</div>")
            else:
                html('<div class="chips"><span class="chip">EMPTY · select targets on the board</span></div>')
        if c[1].button("CLEAR", key="sl_clear", width="stretch", disabled=not len(sl_df)):
            st.session_state["shortlist"] = []
            st.rerun()

# ======================================================================= 5 · ROSTER OPTIONS
with tabs[5]:
    lineup = [p for p in st.session_state["lineup"] if p in set(yr_hit["playerID"])]
    r = st.columns([5.6, 4.4])
    with r[0], panel("Player Pool", "pool", "check up to 9 hitters"):
        c = st.columns([2.6, 1.5, 1], vertical_alignment="center")
        source = c[0].segmented_control("pool", ["MLB", "CLUB", "SHORTLIST"], default="MLB", key="pool_src",
                                        label_visibility="collapsed") or "MLB"
        if c[1].button(f"LOAD {team_id} TOP 9", width="stretch"):
            st.session_state["lineup"] = team_hit.nlargest(9, "PA")["playerID"].tolist()
            st.session_state["pool_ver"] += 1
            st.rerun()
        if c[2].button("CLEAR", key="lu_clear", width="stretch"):
            st.session_state["lineup"] = []
            st.session_state["pool_ver"] += 1
            st.rerun()

        if source == "MLB":
            pool = yr_hit[yr_hit["PA"] >= 200]
        elif source == "SHORTLIST":
            ids = {p for p, y in st.session_state["shortlist"] if y == year}
            pool = yr_hit[yr_hit["playerID"].isin(ids)]
        else:
            pool = team_hit[team_hit["PA"] >= 50]
        pool = pool.sort_values("PEI", ascending=False)
        view = pd.DataFrame({
            "✓": pool["playerID"].isin(lineup).to_numpy(), "Player": pool["fullName"].to_numpy(),
            "Tm": pool["teamID"].to_numpy(), "Pos": pool["Pos"].to_numpy(), "OBP": pool["OBP"].to_numpy(),
            "SLG": pool["SLG"].to_numpy(), "PEI": pool["PEI"].to_numpy(), "Salary": (pool["salary"] / 1e6).to_numpy(),
        }, index=pool["playerID"].to_numpy())
        edited = st.data_editor(
            view, key=f"pool_{year}_{source}_{st.session_state['pool_ver']}", hide_index=True, height=470,
            disabled=["Player", "Tm", "Pos", "OBP", "SLG", "PEI", "Salary"],
            column_config={
                "✓": st.column_config.CheckboxColumn(width="small"),
                "OBP": st.column_config.NumberColumn(format="%.3f"), "SLG": st.column_config.NumberColumn(format="%.3f"),
                "PEI": st.column_config.NumberColumn(format="%.1f"),
                "Salary": st.column_config.NumberColumn("Salary $M", format="%.2f"),
            })
        checked = set(edited.index[edited["✓"]])
        in_view = set(view.index)
        new_lineup = [p for p in lineup if p not in in_view or p in checked] + \
                     [p for p in edited.index if p in checked and p not in lineup]
        if new_lineup != st.session_state["lineup"]:
            st.session_state["lineup"] = new_lineup
        lineup = new_lineup

    lu = yr_hit[yr_hit["playerID"].isin(lineup)].sort_values("OBP", ascending=False)
    over = len(lu) > 9
    if over:
        lu = lu.nlargest(9, "PEI").sort_values("OBP", ascending=False)

    with r[1]:
        with panel("Lineup Card", "card", f"{min(len(lineup), 9)}/9{' · max 9, top PEI used' if over else ''}"):
            html(T.chips(HIT_POS, set(lu["Pos"])))
            rows = "".join(
                f'<div class="r"><span>{i + 1}</span><span>{p["shortName"]}</span><span>{p["Pos"]}</span>'
                f'<span>{T.f3(p["OBP"])}</span><span>{T.money(p["salary"])}</span></div>'
                for i, (_, p) in enumerate(lu.iterrows()))
            html('<div class="card"><div class="r h"><span>#</span><span>PLAYER</span><span>POS</span>'
                 f'<span>OBP</span><span>SALARY</span></div>{rows}</div>')

        with panel("Projection", "proj", f"target {TARGET_RUNS} R · {TARGET_WINS} W"):
            c = st.columns(2)
            ra = c[0].slider("Runs Allowed", 500, 950, int(team["RA"]), 5, key=f"ra_{year}_{franch}")
            budget = c[1].slider("Hitter Budget ($M)", 5, 120, 20, 1)
            if len(lu):
                obp, slg = lineup_stats(lu)
                pred = model.predict_wins(obp, slg, runs_allowed=ra)
                pay = lu["salary"].sum()
                runs, wins = pred["projected_runs"], pred["projected_wins"]
                html('<div class="tiles">'
                     + T.tile("LINEUP OBP", T.f3(obp), f"MLB {T.f3(mlb_obp)}")
                     + T.tile("LINEUP SLG", T.f3(slg), f"{team_id} {T.f3(team['SLG'])}")
                     + T.tile("PAYROLL", T.money(pay), f"budget ${budget}M", pay / (budget * 1e6), gold=pay > budget * 1e6)
                     + T.tile("PROJ. RUNS", f"{runs:.0f}", f"target {TARGET_RUNS}", runs / TARGET_RUNS)
                     + T.tile("RUNS ALLOWED", f"{ra}", f"actual {int(team['RA'])}")
                     + T.tile("PROJ. WINS", f"{wins:.0f}", f"target {TARGET_WINS}", wins / TARGET_WINS, gold=wins >= TARGET_WINS)
                     + "</div>")
                if wins >= TARGET_WINS and pay <= budget * 1e6:
                    html('<div class="verdict ok">PLAYOFF CALIBER · UNDER BUDGET</div>')
                elif wins >= TARGET_WINS:
                    html(f'<div class="verdict no">PLAYOFF CALIBER · OVER BUDGET {T.money(pay - budget * 1e6)}</div>')
                else:
                    html(f'<div class="verdict no">SHORT BY {TARGET_WINS - wins:.0f} WINS</div>')
            else:
                html('<div class="verdict no">SELECT HITTERS IN THE POOL</div>')

# ======================================================================= 6 · REPORTS
with tabs[6]:
    prev_year = year - 1
    has_prev_data = prev_year in set(players["yearID"])
    is_oak_02 = (franch == "OAK" and year == 2002)

    r = st.columns([5.3, 4.7])
    with r[0]:
        panel_title = "Recreating Value in the Aggregate" if is_oak_02 else f"Roster Turnover & Replacement · {team_id}"
        panel_sub = f"case file · {team_id} {prev_year} → {year}" if has_prev_data else f"roster audit · {team_id} {year}"

        with panel(panel_title, "agg", panel_sub):
            use_classic_trio = False
            if is_oak_02:
                t_mode = st.segmented_control(
                    "Turnover Mode",
                    ["Classic Moneyball Trio (Giambi/Damon/Saenz)", "Full Roster Turnover"],
                    default="Classic Moneyball Trio (Giambi/Damon/Saenz)",
                    key="oak02_mode",
                    label_visibility="collapsed"
                )
                use_classic_trio = (t_mode == "Classic Moneyball Trio (Giambi/Damon/Saenz)")

            p_curr = players[players["yearID"] == year].set_index("playerID")

            if use_classic_trio:
                lost_ids = ["giambja01", "damonjo01", "saenzol01"]
                repl_ids = ["giambje01", "hattesc01", "justida01"]
                p_prev = players[players["yearID"] == 2001].set_index("playerID")
                lost_df = p_prev.loc[lost_ids].copy()
                repl_prev_df = p_prev.loc[repl_ids].copy()
                repl_curr_df = p_curr.loc[repl_ids].copy()

                lost_salaries = [p_curr.loc[pid, "salary"] if pid in p_curr.index and p_curr.loc[pid, "salary"] > 0 else p_prev.loc[pid, "salary"] for pid in lost_ids]
                repl_salaries = [repl_curr_df.loc[pid, "salary"] for pid in repl_ids]

                lost_rows = "".join(f'<tr><td>{lost_df.loc[pid, "fullName"]}</td><td class="n">{T.f3(lost_df.loc[pid, "OBP"])}</td>'
                                    f'<td class="n">{T.money(lost_salaries[i])}</td></tr>' for i, pid in enumerate(lost_ids))
                repl_rows = "".join(f'<tr><td>{repl_prev_df.loc[pid, "fullName"]}</td><td class="n">{T.f3(repl_prev_df.loc[pid, "OBP"])}</td>'
                                    f'<td class="n">{T.f3(repl_curr_df.loc[pid, "OBP"])}</td><td class="n">{T.money(repl_salaries[i])}</td></tr>'
                                    for i, pid in enumerate(repl_ids))

                lost_avg = lost_df["OBP"].mean()
                repl_prev_avg = repl_prev_df["OBP"].mean()
                repl_curr_avg = repl_curr_df["OBP"].mean()
                lost_pay_total = sum(lost_salaries)
                repl_pay_total = sum(repl_salaries)

            elif has_prev_data:
                p_prev_all = players[players["yearID"] == prev_year]
                p_curr_all = players[players["yearID"] == year]

                t_prev = p_prev_all[on_team(p_prev_all, team_id) & (p_prev_all["Pos"] != "P") & (p_prev_all["PA"] >= 50)]
                t_curr = p_curr_all[on_team(p_curr_all, team_id) & (p_curr_all["Pos"] != "P") & (p_curr_all["PA"] >= 50)]

                lost_cand = t_prev[~t_prev["playerID"].isin(t_curr["playerID"])].sort_values("PA", ascending=False)
                added_cand = t_curr[~t_curr["playerID"].isin(t_prev["playerID"])].sort_values("PA", ascending=False)

                if len(lost_cand) == 0:
                    lost_cand = t_prev.sort_values("PA", ascending=False)
                if len(added_cand) == 0:
                    added_cand = t_curr.sort_values("PA", ascending=False)

                top_n = min(4, max(len(lost_cand), len(added_cand), 1))
                lost_sample = lost_cand.head(top_n)
                added_sample = added_cand.head(top_n)

                p_prev = p_prev_all.set_index("playerID")

                lost_rows_list = []
                lost_salaries = []
                for _, r_p in lost_sample.iterrows():
                    pid = r_p["playerID"]
                    sal = p_curr.loc[pid, "salary"] if pid in p_curr.index and p_curr.loc[pid, "salary"] > 0 else r_p["salary"]
                    lost_salaries.append(sal)
                    lost_rows_list.append(f'<tr><td>{r_p["fullName"]} ({r_p["Pos"]})</td><td class="n">{T.f3(r_p["OBP"])}</td><td class="n">{T.money(sal)}</td></tr>')

                repl_rows_list = []
                repl_salaries = []
                repl_prev_obps = []
                repl_curr_obps = []
                for _, r_c in added_sample.iterrows():
                    pid = r_c["playerID"]
                    obp_prev = p_prev.loc[pid, "OBP"] if pid in p_prev.index else r_c["OBP"]
                    obp_curr = r_c["OBP"]
                    sal = r_c["salary"]
                    repl_salaries.append(sal)
                    repl_prev_obps.append(obp_prev)
                    repl_curr_obps.append(obp_curr)
                    repl_rows_list.append(f'<tr><td>{r_c["fullName"]} ({r_c["Pos"]})</td><td class="n">{T.f3(obp_prev)}</td><td class="n">{T.f3(obp_curr)}</td><td class="n">{T.money(sal)}</td></tr>')

                lost_rows = "".join(lost_rows_list)
                repl_rows = "".join(repl_rows_list)
                lost_avg = lost_sample["OBP"].mean() if len(lost_sample) else 0.320
                repl_prev_avg = float(np.mean(repl_prev_obps)) if repl_prev_obps else 0.320
                repl_curr_avg = float(np.mean(repl_curr_obps)) if repl_curr_obps else 0.320
                lost_pay_total = sum(lost_salaries)
                repl_pay_total = sum(repl_salaries)

            else:
                lost_avg, repl_prev_avg, repl_curr_avg = team["OBP"], mlb_obp, team["OBP"]
                lost_pay_total = team["Payroll"] * 0.4
                repl_pay_total = team["Payroll"] * 0.3
                lost_rows = "<tr><td colspan='3'>No prior season data available.</td></tr>"
                repl_rows = "<tr><td colspan='4'>Comparing against MLB baseline.</td></tr>"

            c_tables = st.columns(2)
            lbl_yr = f"'{str(prev_year)[-2:]}" if has_prev_data else "PREV"
            lbl_curr = f"'{str(year)[-2:]}"
            with c_tables[0]:
                html(f'<table class="mbt"><tr><th>LOST PLAYERS</th><th class="n">OBP {lbl_yr}</th><th class="n">$ {lbl_curr}</th></tr>'
                     + lost_rows
                     + f'<tr class="hl"><td>AGGREGATE</td><td class="n">{T.f3(lost_avg)}</td><td class="n">{T.money(lost_pay_total)}</td></tr></table>')
            with c_tables[1]:
                html(f'<table class="mbt"><tr><th>ACQUIRED PLAYERS</th><th class="n">OBP {lbl_yr}</th><th class="n">OBP {lbl_curr}</th><th class="n">$ {lbl_curr}</th></tr>'
                     + repl_rows
                     + f'<tr class="hl"><td>AGGREGATE</td><td class="n">{T.f3(repl_prev_avg)}</td><td class="n">{T.f3(repl_curr_avg)}</td><td class="n">{T.money(repl_pay_total)}</td></tr></table>')

            vals = [lost_avg, repl_prev_avg, repl_curr_avg]
            bar_labels = [f"LOST · {prev_year}", f"ADDED · {prev_year} (Exp.)", f"ADDED · {year} (Real.)"]
            fig_turnover = go.Figure(go.Bar(
                x=bar_labels, y=vals, text=[T.f3(v) for v in vals],
                textposition="outside", width=.52, hoverinfo="skip",
                marker=dict(color=["#9dbaa6", "#5f8a72", "#1f4a36"], line=dict(color="#142a20", width=1))
            ))
            fig_turnover.add_hline(y=lost_avg, line=dict(color=T.GOLD_D, dash="dash", width=1.5),
                                   annotation_text=f"Benchmark {T.f3(lost_avg)}", annotation_position="top left",
                                   annotation_font=dict(size=10, color=T.GOLD_D))
            T.style_fig(fig_turnover, 225, margin=dict(l=52, r=12, t=22, b=30))
            y_min = max(0.24, min(vals) * 0.90)
            y_max = max(vals) * 1.08
            fig_turnover.update_yaxes(range=[y_min, y_max], tickformat=".3f")
            st.plotly_chart(fig_turnover, theme=None, width="stretch", config=PLOTLY_CFG)

            obp_diff = repl_curr_avg - lost_avg
            cost_diff = lost_pay_total - repl_pay_total
            diff_color = "#1e3a2f" if obp_diff >= -0.010 else "#6f2f24"
            savings_txt = f"Saved {T.money(cost_diff)}" if cost_diff > 0 else f"Spent {T.money(-cost_diff)}"
            html(f'<div style="background:#e4e1d3; border:1px solid #c9c5b5; border-radius:3px; padding:6px 10px; font-size:12.5px; font-weight:600; color:{diff_color}; margin-top:4px;">'
                 f'TURNOVER DELTA: {obp_diff:+.3f} Aggregate OBP · {savings_txt} in payroll flexibility'
                 f'</div>')

    with r[1]:
        with panel("Executive Season Audit", "audit", f"{franch} · {year} campaign"):
            pay_rank = int(yr_teams["Payroll"].rank(ascending=False).loc[team.name])
            cpw_rank = int(yr_teams["Cost_per_Win"].rank().loc[team.name])
            post_label = ("World Series 🏆" if team["WSWin"] == "Y" else
                          ("Div Champion" if team["DivWin"] == "Y" else
                           ("Wild Card" if team.get("WCWin") == "Y" else "Regular Season")))
            diff_r = int(team["R"] - team["RA"])
            html('<div class="tiles">'
                 + T.tile("RECORD", f"{team['W']}-{team['L']}", f"Win% {team['W']/team['G']:.3f}", team['W']/162)
                 + T.tile("DIV RANK", f"#{team['Rank']}", post_label, gold=team['Rank'] == 1)
                 + T.tile("RUN DIFF", f"{diff_r:+d}", f"{team['R']} R · {team['RA']} RA", (team['R'] - team['RA'] + 200)/400)
                 + T.tile("PYTH. WINS", f"{team['Pythagorean_Wins']:.1f}", f"Luck {team['Win_Diff']:+.1f} W")
                 + T.tile("PAYROLL", T.money(team['Payroll']), f"MLB Rank #{pay_rank} of {len(yr_teams)}")
                 + T.tile("COST / WIN", T.money(team['Cost_per_Win']), f"Value Rank #{cpw_rank} of {len(yr_teams)}", gold=cpw_rank <= 5)
                 + "</div>")

        with panel("Sample Size · Law of Large Numbers", "lln", f"20 simulated seasons · {team_id} {year} baseline (.{int(team['OBP']*1000):03d})"):
            true_obp = st.slider("True OBP Baseline", 0.280, 0.450, float(np.clip(round(team["OBP"], 3), 0.280, 0.450)), 0.005, format="%.3f", key=f"lln_obp_{year}_{franch}")
            pa = np.arange(1, 651)
            rng = np.random.default_rng(42)
            fig_lln = go.Figure()
            for x0, x1, c in ((0, 100, "rgba(111,47,36,.10)"), (100, 300, "rgba(201,162,39,.10)"), (300, 650, "rgba(44,74,59,.10)")):
                fig_lln.add_vrect(x0=x0, x1=x1, fillcolor=c, line_width=0)
            for _ in range(20):
                obs = np.cumsum(rng.binomial(1, true_obp, len(pa))) / pa
                fig_lln.add_trace(go.Scatter(x=pa, y=obs, mode="lines", line=dict(width=1, color="#5f6b64"), opacity=.45, hoverinfo="skip"))
            band = 1.96 * np.sqrt(true_obp * (1 - true_obp) / pa)
            fig_lln.add_trace(go.Scatter(x=pa, y=true_obp + band, mode="lines", line=dict(color=T.GREEN, width=1.5, dash="dot"), hoverinfo="skip"))
            fig_lln.add_trace(go.Scatter(x=pa, y=true_obp - band, mode="lines", line=dict(color=T.GREEN, width=1.5, dash="dot"), hoverinfo="skip"))
            fig_lln.add_hline(y=true_obp, line=dict(color=T.GOLD_D, width=2.5))
            for x, label in ((50, "APR"), (200, "MAY–JUL"), (475, "AUG–SEP")):
                fig_lln.add_annotation(x=x, y=0.58, text=label, showarrow=False, font=dict(size=11, color="#3d4a43"))
            T.style_fig(fig_lln, 260)
            fig_lln.update_xaxes(title="Plate Appearances", range=[1, 650])
            fig_lln.update_yaxes(title="Observed OBP", range=[0.12, 0.62], tickformat=".3f")
            st.plotly_chart(fig_lln, theme=None, width="stretch", config=PLOTLY_CFG)

    with panel("Front-Office Data Export Center", "exp", f"csv datasets · {team_id} & {year} mlb season"):
        c_exp = st.columns(4)
        cols_roster = ["fullName", "Pos", "PA", "AB", "H", "HR", "BB", "OBP", "SLG", "OPS", "PEI", "RC_600", "salary", "Cost_per_RC"]
        roster_out = team_hit[cols_roster] if len(team_hit) else pd.DataFrame(columns=cols_roster)
        cols_standings = ["name", "franchID", "lgID", "divID", "Rank", "W", "L", "R", "RA", "OBP", "SLG", "OPS", "Pythagorean_Wins", "Win_Diff", "Payroll", "Cost_per_Win"]
        standings_out = yr_teams[cols_standings]
        cols_sl = ["fullName", "yearID", "teamID", "Pos", "PA", "OBP", "SLG", "OPS", "PEI", "RC_600", "salary"]
        sl_out = sl_df[cols_sl] if len(sl_df) else pd.DataFrame(columns=cols_sl)
        cols_lu = ["fullName", "Pos", "PA", "OBP", "SLG", "OPS", "PEI", "RC_600", "salary"]
        lu_out = lu[cols_lu] if len(lu) else pd.DataFrame(columns=cols_lu)

        c_exp[0].download_button(f"⬇ {team_id}_{year}_ROSTER.CSV ({len(roster_out)})", roster_out.to_csv(index=False),
                                 f"moneyball_{team_id}_{year}_roster.csv", "text/csv", width="stretch", disabled=not len(roster_out))
        c_exp[1].download_button(f"⬇ {year}_MLB_STANDINGS.CSV ({len(standings_out)})", standings_out.to_csv(index=False),
                                 f"moneyball_{year}_mlb_standings.csv", "text/csv", width="stretch")
        c_exp[2].download_button(f"⬇ SHORTLIST.CSV ({len(sl_out)})", sl_out.to_csv(index=False),
                                 "moneyball_shortlist.csv", "text/csv", width="stretch", disabled=not len(sl_out))
        c_exp[3].download_button(f"⬇ {team_id}_LINEUP_{year}.CSV ({len(lu_out)})", lu_out.to_csv(index=False),
                                 f"moneyball_lineup_{team_id}_{year}.csv", "text/csv", width="stretch", disabled=not len(lu_out))

# ======================================================================= footer
html(f'<div class="ftr">{T.moneyball_wordmark(17)}<span class="t">ANALYTICS. EFFICIENCY. COMPETITIVE ADVANTAGE.</span>'
     f'<span class="by">by daivasukrti9. {T.gemini_icon(14)}</span><img src="{T.logo_url(franch)}"/></div>')
