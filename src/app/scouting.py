"""
scouting.py - Video Scouting Terminal & Strike Zone Matrix
Faithful 1:1 replica of the BATS video scouting terminal from the Moneyball film.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from . import theme as T

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PEOPLE_CSV = os.path.join(BASE_DIR, "data", "People.csv")
PITCHING_PARQUET = os.path.join(BASE_DIR, "data", "processed", "pitching_processed.parquet")

# Curated registry of official MLB IDs for instant zero-latency high-res portraits
KNOWN_MLB_IDS: dict[str, int] = {
    "giambje01": 136605,  # Jeremy Giambi
    "hattesc01": 115608,  # Scott Hatteberg
    "justida01": 116798,  # David Justice
    "tejadmi01": 123173,  # Miguel Tejada
    "chaveer01": 136767,  # Eric Chavez
    "durhara01": 113645,  # Ray Durham
    "penaca01": 276541,   # Carlos Pena
    "ellisma01": 407885,  # Mark Ellis
    "bradfch01": 136268,  # Chad Bradford
    "zitoba01": 217096,   # Barry Zito
    "hudsoti01": 218596,  # Tim Hudson
    "muldema01": 150244,  # Mark Mulder
    "lawrebr01": 400147,  # Brian Lawrence
    "damonjo01": 113028,  # Johnny Damon
    "giambja01": 114739,  # Jason Giambi
    "saenzol01": 121617,  # Olmedo Saenz
    "gantron01": 114569,  # Ron Gant
    "kochbi01": 150116,   # Billy Koch
    "longte01": 150395,   # Terrence Long
    "hernara01": 218595,  # Ramon Hernandez
    "piattad01": 150282,  # Adam Piatt
    "byrneer01": 276534,  # Eric Byrnes
    "jeterde01": 116539,  # Derek Jeter
    "willibe01": 124343,  # Bernie Williams
    "posadjo01": 120691,  # Jorge Posada
    "riverma01": 121257,  # Mariano Rivera
    "ortizda01": 120074,  # David Ortiz
    "ramirma01": 120903,  # Manny Ramirez
    "martipe01": 118377,  # Pedro Martinez
    "bondsba01": 111188,  # Barry Bonds
    "suzukic01": 400085,  # Ichiro Suzuki
    "pujolal01": 405395,  # Albert Pujols
    "rodrial01": 121347,  # Alex Rodriguez
}

_PEOPLE_CACHE: pd.DataFrame | None = None
_PITCHING_CACHE: pd.DataFrame | None = None


def load_people_df() -> pd.DataFrame:
    global _PEOPLE_CACHE
    if _PEOPLE_CACHE is None:
        if os.path.exists(PEOPLE_CSV):
            _PEOPLE_CACHE = pd.read_csv(PEOPLE_CSV, low_memory=False)
        else:
            _PEOPLE_CACHE = pd.DataFrame()
    return _PEOPLE_CACHE


def load_pitching_df() -> pd.DataFrame:
    global _PITCHING_CACHE
    if _PITCHING_CACHE is None:
        if os.path.exists(PITCHING_PARQUET):
            _PITCHING_CACHE = pd.read_parquet(PITCHING_PARQUET)
        else:
            _PITCHING_CACHE = pd.DataFrame()
    return _PITCHING_CACHE


def get_headshot_url(player_id: str) -> str:
    mlb_id = KNOWN_MLB_IDS.get(player_id)
    if mlb_id:
        return f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{mlb_id}/headshot/67/current"
    return "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/generic/headshot/67/current"


def get_player_bio(player_id: str, player_row: pd.Series) -> dict:
    people = load_people_df()
    p_info = {}
    if not people.empty and "playerID" in people.columns:
        match = people[people["playerID"] == player_id]
        if len(match):
            p_info = match.iloc[0].to_dict()

    # Format Height
    h = p_info.get("height", np.nan)
    if pd.notna(h) and float(h) > 0:
        h_str = f"{int(float(h) // 12)}'{int(float(h) % 12):02d}\""
    else:
        h_str = "6'00\""

    # Format Weight
    w = p_info.get("weight", np.nan)
    w_str = f"{int(float(w))} lb." if pd.notna(w) and float(w) > 0 else "195 lb."

    # Handedness
    bats_raw = str(p_info.get("bats") or player_row.get("bats", "R")).upper()
    bats_map = {"R": "Right", "L": "Left", "B": "Both (Switch)"}
    bats_str = bats_map.get(bats_raw, "Right")

    throws_raw = str(p_info.get("throws") or player_row.get("throws", "R")).upper()
    throws_map = {"R": "Right", "L": "Left"}
    throws_str = throws_map.get(throws_raw, "Right")

    # Born
    by = p_info.get("birthYear", player_row.get("birthYear", np.nan))
    b_city = p_info.get("birthCity", "")
    b_state = p_info.get("birthState", "")
    loc_str = f" in {b_city}, {b_state}" if b_city else ""
    born_str = f"{int(float(by))}{loc_str}" if pd.notna(by) else "1975"

    debut_str = str(p_info.get("debut", "1998-09-01"))
    if pd.isna(debut_str) or debut_str == "nan":
        debut_str = "1998-09-01"

    salary = float(player_row.get("salary", 0))
    cost_rc = float(player_row.get("Cost_per_RC", 0))

    return {
        "fullName": player_row.get("fullName", "Unknown Player"),
        "Pos": player_row.get("Pos", "DH"),
        "bats": bats_str,
        "throws": throws_str,
        "height": h_str,
        "weight": w_str,
        "born": born_str,
        "debut": debut_str,
        "salary_str": T.money(salary) if salary > 0 else "Pre-Arbitration",
        "cost_rc_str": f"${cost_rc:,.0f}" if cost_rc > 0 else "N/A",
        "headshot_url": get_headshot_url(player_id),
        "playerID": player_id,
    }


def get_bio_card_html(bio: dict) -> str:
    """
    Renders the Ficha Técnica card with player photo, halftone scanline texture, and bio table.
    Styled after screenshot 20261101_113155.jpg.
    """
    return f"""
    <div class="bio-card">
        <div class="bio-photo-wrap">
            <img class="bio-photo" src="{bio['headshot_url']}" alt="{bio['fullName']}"/>
        </div>
        <div style="flex:1; min-width:0; overflow:hidden;">
            <div style="font-family:'Barlow Semi Condensed', sans-serif; font-size:20px; font-weight:700; color:#142a20; letter-spacing:.04em; line-height:1.1; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {bio['fullName'].upper()}
            </div>
            <table class="bio-table">
                <tr><td class="k">Position:</td><td class="v"><b>{bio['Pos']}</b></td></tr>
                <tr><td class="k">Bats:</td><td class="v">{bio['bats']}</td></tr>
                <tr><td class="k">Throws:</td><td class="v">{bio['throws']}</td></tr>
                <tr><td class="k">Height:</td><td class="v">{bio['height']}</td></tr>
                <tr><td class="k">Weight:</td><td class="v">{bio['weight']}</td></tr>
                <tr><td class="k">Born:</td><td class="v">{bio['born']}</td></tr>
                <tr><td class="k">Debut:</td><td class="v">{bio['debut']}</td></tr>
                <tr><td class="k">Salary:</td><td class="v"><b>{bio['salary_str']}</b></td></tr>
                <tr><td class="k">$/RC:</td><td class="v">{bio['cost_rc_str']}</td></tr>
            </table>
        </div>
    </div>
    """


def generate_scouting_pitches(player_row: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Generates authentic pitch coordinates and event profiles matching the movie screen.
    """
    rng = np.random.default_rng(seed + int(player_row.get("H", 40)))

    pitches = []
    ptypes = ["FASTBALL", "SLIDER", "CHANGEUP", "CURVEBALL"]
    p_weights = [0.65, 0.20, 0.10, 0.05]

    hrs = int(player_row.get("HR", 5))
    hits = int(player_row.get("H", 40)) - hrs
    so = int(player_row.get("SO", 30))
    bb = int(player_row.get("BB", 30))

    n_hr = min(max(hrs // 2, 1), 4)
    n_h = min(max(hits // 4, 3), 7)
    n_s = 10
    n_b = 8
    n_f = 4

    # 1. Home Runs (Sweet spot: middle-in / middle-out, inside the zone)
    for _ in range(n_hr):
        x = rng.uniform(-0.55, 0.55)
        y = rng.uniform(2.0, 3.1)
        ptype = rng.choice(["FASTBALL", "SLIDER"], p=[0.75, 0.25])
        mph = int(rng.uniform(91, 95)) if ptype == "FASTBALL" else int(rng.uniform(82, 86))
        count = rng.choice(["2-1", "3-1", "1-0", "2-0"])
        desc = f"HR · {int(rng.uniform(390, 435))} ft RF (Exit Velo {int(rng.uniform(103, 109))} MPH)"
        pitches.append({"x": x, "y": y, "pitch_type": ptype, "mph": mph, "result": "HR", "count": count, "desc": desc, "inn": rng.choice(["1-3", "4-6", "7-9"]), "outs": rng.choice([0, 1, 2])})

    # 2. Hits (Singles / Doubles inside or clipping edge of zone)
    for _ in range(n_h):
        x = rng.uniform(-0.75, 0.75)
        y = rng.uniform(1.65, 3.35)
        ptype = rng.choice(ptypes, p=p_weights)
        mph = int(rng.uniform(90, 95)) if ptype == "FASTBALL" else int(rng.uniform(80, 86))
        count = rng.choice(["0-1", "1-1", "2-1", "1-2"])
        desc = rng.choice(["Single · Line drive LF", "Single · Sharp grounder 5.5 hole", "Double · Left-center gap", "Single · Up the middle"])
        pitches.append({"x": x, "y": y, "pitch_type": ptype, "mph": mph, "result": "H", "count": count, "desc": desc, "inn": rng.choice(["1-3", "4-6", "7-9"]), "outs": rng.choice([0, 1, 2])})

    # 3. Strikes (Called and swinging on edges/chase)
    for _ in range(n_s):
        in_zone = rng.random() < 0.65
        if in_zone:
            x = rng.uniform(-0.80, 0.80)
            y = rng.uniform(1.55, 3.45)
        else:
            x = rng.choice([rng.uniform(-1.2, -0.85), rng.uniform(0.85, 1.25)])
            y = rng.uniform(1.3, 3.7)
        ptype = rng.choice(ptypes, p=p_weights)
        mph = int(rng.uniform(92, 96)) if ptype == "FASTBALL" else int(rng.uniform(81, 87))
        count = rng.choice(["0-1", "0-2", "1-2", "2-2"])
        desc = rng.choice(["Swinging Strike · Whiff", "Called Strike · Paint on corner", "Foul Tip into mitt", "Swinging Strike 3"])
        pitches.append({"x": x, "y": y, "pitch_type": ptype, "mph": mph, "result": "S", "count": count, "desc": desc, "inn": rng.choice(["1-3", "4-6", "7-9"]), "outs": rng.choice([0, 1, 2])})

    # 4. Balls (Outside the zone)
    for _ in range(n_b):
        side = rng.choice(["low", "high", "wide_left", "wide_right"])
        if side == "low":
            x, y = rng.uniform(-0.9, 0.9), rng.uniform(0.9, 1.45)
        elif side == "high":
            x, y = rng.uniform(-0.9, 0.9), rng.uniform(3.55, 4.1)
        elif side == "wide_left":
            x, y = rng.uniform(-1.5, -0.9), rng.uniform(1.6, 3.4)
        else:
            x, y = rng.uniform(0.9, 1.5), rng.uniform(1.6, 3.4)
        ptype = rng.choice(ptypes, p=p_weights)
        mph = int(rng.uniform(91, 95)) if ptype == "FASTBALL" else int(rng.uniform(80, 85))
        count = rng.choice(["1-0", "2-0", "3-0", "2-1", "3-1"])
        desc = "Ball outside the zone · Discipline hold"
        pitches.append({"x": x, "y": y, "pitch_type": ptype, "mph": mph, "result": "B", "count": count, "desc": desc, "inn": rng.choice(["1-3", "4-6", "7-9"]), "outs": rng.choice([0, 1, 2])})

    # 5. Fouls
    for _ in range(n_f):
        x = rng.uniform(-0.85, 0.85)
        y = rng.uniform(1.6, 3.4)
        ptype = rng.choice(ptypes, p=p_weights)
        mph = int(rng.uniform(92, 95))
        count = rng.choice(["0-1", "1-2", "2-2"])
        desc = "Foul ball into grandstand"
        pitches.append({"x": x, "y": y, "pitch_type": ptype, "mph": mph, "result": "F", "count": count, "desc": desc, "inn": rng.choice(["1-3", "4-6", "7-9"]), "outs": rng.choice([0, 1, 2])})

    return pd.DataFrame(pitches)


def render_strike_zone_fig(pitches_df: pd.DataFrame, bats: str = "Left") -> go.Figure:
    """
    Renders the 3x3 Strike Zone and Batter silhouette matching the Moneyball film screenshot.
    """
    fig = go.Figure()

    # Strike zone outer box: x [-0.85, 0.85], y [1.5, 3.5]
    fig.add_shape(
        type="rect", x0=-0.85, x1=0.85, y0=1.5, y1=3.5,
        line=dict(color="#1f2520", width=2.5),
        fillcolor="#d7d2be",
        layer="below"
    )

    # 3x3 grid lines
    dx = 1.70 / 3.0
    dy = 2.00 / 3.0
    for i in [1, 2]:
        vx = -0.85 + i * dx
        fig.add_shape(type="line", x0=vx, x1=vx, y0=1.5, y1=3.5,
                      line=dict(color="#666f65", width=1.2, dash="dash"), layer="below")
        vy = 1.5 + i * dy
        fig.add_shape(type="line", x0=-0.85, x1=0.85, y0=vy, y1=vy,
                      line=dict(color="#666f65", width=1.2, dash="dash"), layer="below")

    # Home Plate below
    fig.add_shape(
        type="path",
        path="M -0.70 1.05 L 0.70 1.05 L 0.70 1.25 L 0 1.42 L -0.70 1.25 Z",
        line=dict(color="#39423b", width=1.5),
        fillcolor="#f2f0e6",
        layer="below"
    )

    # Determine batting orientation: Left (L), Right (R), or Both/Switch (S)
    bats_str = str(bats).strip().upper()
    is_left = ("LEFT" in bats_str) or (bats_str == "L")
    is_right = ("RIGHT" in bats_str) or (bats_str == "R")
    is_switch = ("BOTH" in bats_str) or ("SWITCH" in bats_str) or (bats_str == "B") or (not is_left and not is_right)

    # Catcher's view:
    # Left side (x = -1.38) = Right-handed batter box (bat labeled R)
    # Right side (x = 1.38) = Left-handed batter box (bat labeled L)
    bat_configs = [
        {"bx": -1.38, "letter": "R", "label": "RIGHT", "active": is_right or is_switch},
        {"bx":  1.38, "letter": "L", "label": "LEFT",  "active": is_left or is_switch},
    ]

    knob_w, handle_w, barrel_bot_w, barrel_top_w = 0.08, 0.045, 0.085, 0.125

    for cfg in bat_configs:
        bx = cfg["bx"]
        act = cfg["active"]
        let = cfg["letter"]
        lbl = cfg["label"]

        if act:
            wood_fill = "#c49a52"
            wood_border = "#2d1f10"
            grip_fill = "#ded8c8"
            grip_line = "#6e563b"
            badge_fill = "#142a20"
            badge_border = "#f3d97a"
            badge_text = "#f3d97a"
            label_color = "#142a20"
            opacity = 1.0
            b_width = 2.5
        else:
            wood_fill = "#9fa69b"
            wood_border = "#687065"
            grip_fill = "#b8bfb4"
            grip_line = "#7d857a"
            badge_fill = "#6f756c"
            badge_border = "#8e968b"
            badge_text = "#ced4cb"
            label_color = "#757c72"
            opacity = 0.30
            b_width = 1.5

        # 1. Bat Body Path
        bat_path = (f"M {bx-knob_w} 1.25 L {bx+knob_w} 1.25 L {bx+handle_w} 1.36 "
                    f"L {bx+handle_w} 2.18 L {bx+barrel_bot_w} 2.60 L {bx+barrel_top_w} 3.48 "
                    f"Q {bx} 3.60 {bx-barrel_top_w} 3.48 L {bx-barrel_bot_w} 2.60 "
                    f"L {bx-handle_w} 2.18 L {bx-handle_w} 1.36 Z")
        fig.add_shape(type="path", path=bat_path, fillcolor=wood_fill, opacity=opacity,
                      line=dict(color=wood_border, width=1.8), layer="below")

        # 2. Grip tape wrap on handle
        grip_path = (f"M {bx-handle_w} 1.38 L {bx+handle_w} 1.38 "
                     f"L {bx+handle_w} 2.15 L {bx-handle_w} 2.15 Z")
        fig.add_shape(type="path", path=grip_path, fillcolor=grip_fill, opacity=opacity,
                      line=dict(color=wood_border, width=1.0), layer="below")

        # 3. Grip rings
        for gy in [1.58, 1.78, 1.98]:
            fig.add_shape(type="line", x0=bx-handle_w, x1=bx+handle_w,
                          y0=gy, y1=gy+0.03, opacity=opacity,
                          line=dict(color=grip_line, width=1.0), layer="below")

        # 4. Circular Badge with R / L Letter crowning the bat
        badge_y = 3.92
        fig.add_shape(type="circle", x0=bx-0.25, x1=bx+0.25,
                      y0=badge_y-0.25, y1=badge_y+0.25, fillcolor=badge_fill, opacity=opacity,
                      line=dict(color=badge_border, width=b_width), layer="above")
        fig.add_annotation(x=bx, y=badge_y, text=f"<b>{let}</b>", showarrow=False,
                           font=dict(family="IBM Plex Mono, monospace", size=19, color=badge_text), opacity=opacity)

        # 5. Handedness label below bat
        status_suffix = " (ACTIVE)" if act else ""
        fig.add_annotation(x=bx, y=1.05, text=f"<b>{lbl}{status_suffix}</b>", showarrow=False,
                           font=dict(family="IBM Plex Mono, monospace", size=9, color=label_color), opacity=opacity)

    # Plot Pitch Dots
    if len(pitches_df):
        for _, p in pitches_df.iterrows():
            res = p["result"]
            if res == "HR":
                bg, border, txt = "#c9a227", "#614902", "HR"
            elif res == "H":
                bg, border, txt = "#2e7d32", "#134217", "H"
            elif res == "S":
                bg, border, txt = "#c62828", "#5c0808", "S"
            elif res == "F":
                bg, border, txt = "#5e6660", "#2b302c", "F"
            else:
                bg, border, txt = "#00838f", "#00474f", "B"

            fig.add_trace(go.Scatter(
                x=[p["x"]], y=[p["y"]],
                mode="markers+text",
                text=[txt],
                textposition="middle center",
                textfont=dict(color="#ffffff", size=9.5, family="IBM Plex Mono, monospace"),
                marker=dict(size=23, color=bg, line=dict(color=border, width=1.5)),
                customdata=[[p["pitch_type"], p["mph"], p["count"], p["desc"]]],
                hovertemplate="<b>%{customdata[0]} · %{customdata[1]} MPH</b><br>"
                              "Count: %{customdata[2]}<br>"
                              "Result: %{customdata[3]}<extra></extra>"
            ))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#dcdacd",
        plot_bgcolor="#cbcfc3",
        showlegend=False,
        hoverlabel=dict(bgcolor="#10251c", font_color="#f5f3ea", font_family="IBM Plex Mono"),
        xaxis=dict(range=[-1.9, 1.9], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0.8, 4.3], showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def render_diamond_fig(sb: int = 12, cs: int = 3) -> go.Figure:
    """
    Renders the baseball diamond for Baserunning & Speed Scouting.
    """
    fig = go.Figure()
    pts = [(0, 0), (1, 1), (0, 2), (-1, 1), (0, 0)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#757063", width=3), hoverinfo="skip"))

    # Pitcher Mound
    fig.add_shape(type="circle", x0=-0.18, x1=0.18, y0=0.82, y1=1.18, fillcolor="#b8b4a2", line=dict(color="#5a5649", width=1.5))
    fig.add_shape(type="rect", x0=-0.08, x1=0.08, y0=0.97, y1=1.03, fillcolor="#f2f0e6", line_width=0)

    # Bases
    bases = [("HOME", 0, 0), ("1B", 1, 1), ("2B", 0, 2), ("3B", -1, 1)]
    for name, bx, by in bases:
        fig.add_trace(go.Scatter(
            x=[bx], y=[by], mode="markers+text", text=[name], textposition="middle center",
            marker=dict(size=26, color="#f5f3ea", line=dict(color="#2c4a3b", width=2.5)),
            textfont=dict(size=9.5, family="IBM Plex Mono", color="#1e3a2f", weight="bold"), hoverinfo="skip"
        ))

    # Steal Lane to 2nd Base
    success_rate = (sb / (sb + cs) * 100) if (sb + cs) > 0 else 0
    fig.add_trace(go.Scatter(
        x=[0.5], y=[1.5], mode="markers+text",
        text=[f"2B STEALS: {sb} SB / {cs} CS ({success_rate:.0f}%)"],
        textposition="top right",
        marker=dict(size=14, color="#2e7d32" if success_rate >= 75 else "#c62828"),
        textfont=dict(size=11, color="#1e3a2f", family="IBM Plex Mono", weight="bold"),
        hoverinfo="skip"
    ))

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#dcdacd", plot_bgcolor="#cbcfc3", showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.25, 2.25])
    )
    return fig


def get_pitch_breakdown_html(pitches_df: pd.DataFrame, pa: int = 156) -> str:
    """
    Renders the Pitch Breakdown table matching screenshot 20261101_104216.jpg.
    """
    mult = max(pa / 156.0, 0.4)
    data = [
        ("FASTBALL", int(482 * mult), 77, int(147 * mult), int(335 * mult), int(239 * mult)),
        ("SLIDER", int(122 * mult), 19, int(57 * mult), int(65 * mult), int(42 * mult)),
        ("CHANGEUP", int(21 * mult), 3, int(12 * mult), int(9 * mult), int(7 * mult)),
        ("CURVEBALL", int(8 * mult), 1, int(5 * mult), int(3 * mult), int(2 * mult)),
        ("PITCHOUT", int(1 * mult), 0, int(1 * mult), 0, 0),
    ]
    rows = "".join(
        f"<tr><td><b>{name}</b></td><td class='n'>{tot}</td><td class='n'>{pct}%</td>"
        f"<td class='n'>{b}</td><td class='n'>{s}</td><td class='n'>{sw}</td></tr>"
        for name, tot, pct, b, s, sw in data
    )
    return f"""
    <table class="mbt" style="font-size:12px; margin-bottom:8px;">
        <tr><th>PITCH TYPE</th><th class="n">Total</th><th class="n">Pct</th><th class="n">Ball</th><th class="n">Str</th><th class="n">Sw</th></tr>
        {rows}
    </table>
    """


def get_boxed_statistics_html(row: pd.Series) -> str:
    """
    Renders the inset beveled Statistics boxes matching screenshot 20261101_104216.jpg & 20261101_104234.jpg.
    """
    g = int(row.get("G", 40))
    ab = int(row.get("AB", 153))
    h = int(row.get("H", 44))
    b2 = int(row.get("2B", 4))
    b3 = int(row.get("3B", 1))
    hr = int(row.get("HR", 6))
    rbi = int(row.get("RBI", 22))
    bb = int(row.get("BB", 6))
    so = int(row.get("SO", 33))
    hbp = int(row.get("HBP", 1))
    sac = int(row.get("SF", 0) + row.get("SH", 1))
    dp = int(row.get("GIDP", 14))
    tpa = int(row.get("PA", ab + bb + hbp + sac))

    ba = f"{h / ab:.3f}" if ab > 0 else ".000"
    sp = f"{row.get('SLG', 0.446):.3f}"
    obp = f"{row.get('OBP', 0.414):.3f}"

    gb = max(int(ab * 0.47), 1)
    fb = max(int(ab * 0.32), 1)
    ld = max(int(ab * 0.15), 1)
    pu = max(int(ab * 0.06), 1)

    items = [
        ("G", g), ("SAC", sac), ("FB", fb),
        ("AB", ab), ("DP", dp), ("GB", gb),
        ("H", h), ("HBP", hbp), ("LD", ld),
        ("2B", b2), ("TPA", tpa), ("PU", pu),
        ("3B", b3), ("BA", ba), ("Hard", "38%"),
        ("HR", hr), ("SP", sp), ("Med", "45%"),
        ("RBI", rbi), ("OBP", obp), ("Soft", "17%"),
        ("BB", bb), ("OPS", f"{row.get('OPS', 0.860):.3f}"), ("RH", 1),
        ("K", so),
    ]

    cells = "".join(
        f'<div class="mb-stat-item"><span class="k">{k}:</span><span class="v">{v}</span></div>'
        for k, v in items
    )
    return f"""
    <div style="background:#cfccc0; border:2px solid #8e8b7d; padding:8px 10px; border-radius:3px;">
        <div style="font-size:11.5px; font-weight:700; color:#14251d; letter-spacing:.12em; margin-bottom:6px; background:#f3d97a; display:inline-block; padding:1px 6px;">STATISTICS</div>
        <div class="mb-stat-grid">{cells}</div>
    </div>
    """


def get_pitcher_arsenal_html(player_id: str) -> str:
    """
    Renders pitching repertoire and velocity metrics for pitchers (e.g. Chad Bradford, Barry Zito).
    """
    is_bradford = (player_id == "bradfch01")
    is_zito = (player_id == "zitoba01")

    if is_bradford:
        # Chad Bradford submarine profile
        data = [
            ("SINKER (SUBMARINE)", "85 MPH", "58%", "68%", "21%", "Extreme GB Rate (78%)"),
            ("SLIDER", "78 MPH", "26%", "64%", "31%", "Sharp ankle sweep"),
            ("CHANGEUP", "75 MPH", "16%", "59%", "26%", "Deceptive release"),
        ]
    elif is_zito:
        # Barry Zito 12-6 curveball profile
        data = [
            ("CURVEBALL (12-6)", "74 MPH", "42%", "69%", "39%", "Signature vertical drop"),
            ("4-SEAM FASTBALL", "89 MPH", "41%", "64%", "19%", "Painted high in zone"),
            ("CHANGEUP", "79 MPH", "17%", "62%", "28%", "Late tumble"),
        ]
    else:
        data = [
            ("4-SEAM FASTBALL", "92 MPH", "54%", "65%", "22%", "Primary heater"),
            ("SLIDER / CUTTER", "85 MPH", "28%", "62%", "30%", "Glove-side break"),
            ("CHANGEUP", "82 MPH", "18%", "59%", "25%", "Arm-side fade"),
        ]

    rows = "".join(
        f"<tr><td><b>{name}</b></td><td class='n'>{velo}</td><td class='n'>{pct}</td>"
        f"<td class='n'>{strk}</td><td class='n'>{whiff}</td><td>{notes}</td></tr>"
        for name, velo, pct, strk, whiff, notes in data
    )
    return f"""
    <table class="mbt" style="font-size:12px; margin-bottom:8px;">
        <tr><th>ARSENAL PITCH</th><th class="n">Velocity</th><th class="n">Usage</th><th class="n">Strike%</th><th class="n">Whiff%</th><th>Action Profile</th></tr>
        {rows}
    </table>
    """


def get_pitcher_statistics_html(player_id: str, year: int) -> str:
    """
    Renders pitching stats from the pitching cache.
    """
    pdf = load_pitching_df()
    match = pd.DataFrame()
    if not pdf.empty:
        match = pdf[(pdf["playerID"] == player_id) & (pdf["yearID"] == year)]

    if len(match):
        r = match.iloc[0]
        w, l, era, ip = int(r["W"]), int(r["L"]), f"{r['ERA']:.2f}", f"{r['IP']:.1f}"
        so, bb, whip, fip = int(r["SO"]), int(r["BB"]), f"{r['WHIP']:.2f}", f"{r['FIP']:.2f}"
        k9, bb9 = f"{r['K9']:.1f}", f"{r['BB9']:.1f}"
        gs = int(r.get("GS", 0))
        sv = int(r.get("SV", 0))
    else:
        # Default placeholder for position players pitching or missing
        w, l, era, ip, so, bb, whip, fip, k9, bb9, gs, sv = 4, 2, "3.11", "75.1", 56, 14, "1.15", "2.82", "6.7", "1.7", 0, 2

    items = [
        ("W", w), ("ERA", era), ("K/9", k9),
        ("L", l), ("WHIP", whip), ("BB/9", bb9),
        ("IP", ip), ("FIP", fip), ("GS", gs),
        ("SO", so), ("SV", sv), ("GB%", "64%"),
        ("BB", bb), ("LOB%", "78%"), ("Hard%", "22%"),
    ]
    cells = "".join(
        f'<div class="mb-stat-item"><span class="k">{k}:</span><span class="v">{v}</span></div>'
        for k, v in items
    )
    return f"""
    <div style="background:#cfccc0; border:2px solid #8e8b7d; padding:8px 10px; border-radius:3px;">
        <div style="font-size:11.5px; font-weight:700; color:#14251d; letter-spacing:.12em; margin-bottom:6px; background:#f3d97a; display:inline-block; padding:1px 6px;">PITCHING AUDIT</div>
        <div class="mb-stat-grid">{cells}</div>
    </div>
    """


def get_baserunning_statistics_html(player_row: pd.Series) -> str:
    """
    Renders baserunning efficiency stats and the Moneyball 75% break-even rule.
    """
    sb = int(player_row.get("SB", 3))
    cs = int(player_row.get("CS", 1))
    attempts = sb + cs
    rate = (sb / attempts * 100) if attempts > 0 else 0.0
    gidp = int(player_row.get("GIDP", 6))
    runs = int(player_row.get("R", 35))

    items = [
        ("SB", sb), ("ATT", attempts), ("R", runs),
        ("CS", cs), ("SB%", f"{rate:.1f}%"), ("XBT%", "42%"),
        ("GIDP", gidp), ("UBR", "+1.2"), ("S-Score", "62"),
    ]
    cells = "".join(
        f'<div class="mb-stat-item"><span class="k">{k}:</span><span class="v">{v}</span></div>'
        for k, v in items
    )

    verdict_color = "#1e3a2f" if rate >= 75 or attempts == 0 else "#6f2f24"
    verdict_txt = "EFFICIENT (EXCEEDS 75% BREAK-EVEN)" if rate >= 75 else ("CONSERVATIVE DISCIPLINE (LOW ATTEMPTS)" if attempts <= 3 else "INEFFICIENT (BELOW 75% THRESHOLD)")

    return f"""
    <div style="background:#cfccc0; border:2px solid #8e8b7d; padding:8px 10px; border-radius:3px;">
        <div style="font-size:11.5px; font-weight:700; color:#14251d; letter-spacing:.12em; margin-bottom:6px; background:#f3d97a; display:inline-block; padding:1px 6px;">BASERUNNING AUDIT</div>
        <div class="mb-stat-grid">{cells}</div>
        <div style="margin-top:8px; font-size:11px; font-weight:700; color:{verdict_color}; background:#e4e1d3; padding:4px 6px; border:1px solid #c9c5b5;">
            SABERMETRIC VERDICT: {verdict_txt}
        </div>
    </div>
    """
