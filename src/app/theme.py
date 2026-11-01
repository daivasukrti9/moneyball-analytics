"""Tema visual "Baseball Operations 2002": pizarra crema, barras verde bosque y resaltado dorado."""
import base64
import os

import pandas as pd

GREEN_D = "#1e3a2f"
GREEN = "#2c4a3b"
GREEN_M = "#4f7a63"
GREEN_L = "#9dbaa6"
CREAM = "#eceadf"
PAPER = "#f4f2e8"
GOLD = "#f3d97a"
GOLD_D = "#c9a227"
INK = "#1f2a24"
GRAY = "#7d7d74"
GRID = "#d3d0c2"

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))

# franchID (Lahman) -> abreviatura de logos ESPN
LOGO_ABBR = {
    "ANA": "laa", "ARI": "ari", "ATL": "atl", "BAL": "bal", "BOS": "bos", "CHC": "chc",
    "CHW": "chw", "CIN": "cin", "CLE": "cle", "COL": "col", "DET": "det", "FLA": "mia",
    "HOU": "hou", "KCR": "kc", "LAD": "lad", "MIL": "mil", "MIN": "min", "NYM": "nym",
    "NYY": "nyy", "OAK": "oak", "PHI": "phi", "PIT": "pit", "SDP": "sd", "SEA": "sea",
    "SFG": "sf", "STL": "stl", "TBD": "tb", "TEX": "tex", "TOR": "tor", "WSN": "wsh",
}


def logo_url(franch_id: str) -> str:
    abbr = LOGO_ABBR.get(franch_id)
    if abbr is None:
        return "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png"
    return f"https://a.espncdn.com/i/teamlogos/mlb/500/{abbr}.png"


def moneyball_wordmark(size: int = 20) -> str:
    """Logo de la película: usa assets/moneyball_logo.png si existe; si no, un wordmark tipográfico."""
    for name in ("moneyball_logo.png", "moneyball_logo.jpg", "moneyball_logo.svg"):
        path = os.path.join(ASSETS_DIR, name)
        if os.path.exists(path):
            mime = "image/svg+xml" if name.endswith(".svg") else f"image/{name.rsplit('.', 1)[1].replace('jpg', 'jpeg')}"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f'<img class="mbw-img" style="height:{size * 1.7:.0f}px" src="data:{mime};base64,{b64}"/>'
    return f'<span class="mbw" style="font-size:{size}px">MONEYBALL</span>'


def gemini_icon(size: int = 14) -> str:
    """Icono sutil oficial de Google Gemini (estrella de 4 puntas cóncava) con degradado."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'style="vertical-align:-2px; margin-left:4px; display:inline-block; filter:drop-shadow(0 0 3px rgba(186,155,251,0.5));">'
        f'<defs><linearGradient id="geminiSparkleGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="#7ba1f7"/>'
        f'<stop offset="50%" stop-color="#ba9bfb"/>'
        f'<stop offset="100%" stop-color="#f3d97a"/>'
        f'</linearGradient></defs>'
        f'<path d="M12 0C12 6.627 6.627 12 0 12C6.627 12 12 17.373 12 24C12 17.373 17.373 12 24 12C17.373 12 12 6.627 12 0Z" '
        f'fill="url(#geminiSparkleGrad)"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------- formato
def f3(x) -> str:
    """.383 en lugar de 0.383 (notación clásica de béisbol)."""
    if pd.isna(x):
        return "—"
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0") else s


def money(x) -> str:
    if pd.isna(x) or x <= 0:
        return "—"
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"


# ---------------------------------------------------------------- componentes HTML
def panel_header(title: str, sub: str | None = None) -> str:
    sub_html = f"<small>{sub}</small>" if sub else ""
    return f'<div class="ph"><span>{title}</span>{sub_html}</div>'


def rank_table(df: pd.DataFrame, show_team: bool = False, highlight: int = 0) -> str:
    head = "<tr><th>Rk</th><th>Player</th>" + ("<th>Tm</th>" if show_team else "") + \
           '<th>Pos</th><th class="n">PEI</th><th class="n">OBP</th><th class="n">SLG</th><th class="n">OPS</th></tr>'
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        cls = ' class="hl"' if i == highlight else ""
        team = f"<td>{r['teamID'].split('/')[-1]}</td>" if show_team else ""
        rows.append(
            f"<tr{cls}><td>{i + 1}</td><td>{r['shortName']}</td>{team}<td>{r['Pos']}</td>"
            f'<td class="n">{r["PEI"]:.1f}</td><td class="n">{f3(r["OBP"])}</td>'
            f'<td class="n">{f3(r["SLG"])}</td><td class="n">{f3(r["OPS"])}</td></tr>'
        )
    return f'<table class="mbt">{head}{"".join(rows)}</table>'


def value_list(items: list[tuple[str, str]], highlight: int = 0) -> str:
    rows = "".join(
        f'<div class="r{" hl" if i == highlight else ""}"><span>{i + 1}.&nbsp; {k}</span><span>{v}</span></div>'
        for i, (k, v) in enumerate(items)
    )
    return f'<div class="lo">{rows}</div>'


def bullets(items: list[str]) -> str:
    return '<ul class="bl">' + "".join(f"<li><span>{t}</span></li>" for t in items) + "</ul>"


def kv_rows(items: list[tuple[str, str]]) -> str:
    return '<div class="kv">' + "".join(f'<div class="r"><span>{k}</span><b>{v}</b></div>' for k, v in items) + "</div>"


def chips(labels: list[str], active: set[str]) -> str:
    return '<div class="chips">' + "".join(
        f'<span class="chip{" on" if l in active else ""}">{l}</span>' for l in labels) + "</div>"


def tile(label: str, value: str, sub: str = "", pct: float | None = None, gold: bool = False) -> str:
    bar = ""
    if pct is not None:
        bar = f'<div class="bar"><i class="{"gold" if gold else ""}" style="width:{max(0, min(pct, 1)) * 100:.0f}%"></i></div>'
    return f'<div class="tile"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div>{bar}</div>'


def style_fig(fig, height: int = 300, margin: dict | None = None):
    fig.update_layout(
        height=height,
        margin=margin or dict(l=58, r=16, t=10, b=44),
        paper_bgcolor=CREAM,
        plot_bgcolor=PAPER,
        font=dict(family="IBM Plex Mono, monospace", size=12, color=INK),
        showlegend=False,
        hoverlabel=dict(bgcolor=GREEN_D, font_color="#f1efe4", font_family="IBM Plex Mono, monospace"),
    )
    axis = dict(gridcolor=GRID, linecolor="#8d8a7c", zeroline=False, showline=True, mirror=True,
                ticks="outside", tickcolor="#8d8a7c")
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Barlow+Semi+Condensed:wght@500;600;700&display=swap');

html, body, .stApp, .stMarkdown, p, label, input, textarea, button { font-family: 'IBM Plex Mono', monospace; }
.stApp { background: radial-gradient(ellipse at center, #2b2e2a 0%, #101210 100%); }
header[data-testid="stHeader"] { display: none; }
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

/* Monitor / pantalla */
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1520px; padding: 0 0 0 0 !important; margin: 22px auto 26px auto;
  background: #dcdacd; border: 12px solid #2b2d2a; border-radius: 10px;
  box-shadow: 0 0 0 2px #3c3f3b, 0 30px 70px rgba(0,0,0,.65), inset 0 0 60px rgba(0,0,0,.10);
}
.stTabs { padding: 0 16px; }

/* Header */
.st-key-hdr { background: linear-gradient(180deg, #27493b 0%, #1b3a2e 100%);
  border-bottom: 3px solid #10251c; padding: 14px 26px 12px 26px; }
.st-key-hdr img.tlogo { height: 74px; filter: drop-shadow(0 2px 2px rgba(0,0,0,.4)); }
.hdr-title { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 700; font-size: 36px;
  letter-spacing: .05em; color: #f1efe4; line-height: 1; }
.hdr-sub { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 500; font-size: 19px;
  letter-spacing: .07em; color: #d4d9cb; margin-top: 7px; }
.hdr-tag { font-size: 13px; letter-spacing: .16em; color: #cfd6c6; margin-bottom: 2px; }
.hdr-date { text-align: right; color: #e8e6da; font-size: 15px; line-height: 1.55; letter-spacing: .04em; }
.st-key-hdr label p { color: #a9bcae !important; font-size: 10.5px !important; letter-spacing: .16em; }
.st-key-hdr [data-baseweb="select"] > div { background: #2f5242 !important; border: 1px solid #5a7d69 !important;
  min-height: 32px; border-radius: 2px; }
.st-key-hdr [data-baseweb="select"] * { color: #f1efe4 !important; }
.st-key-hdr [data-baseweb="select"] svg { fill: #f1efe4 !important; }
.mbw { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 700; letter-spacing: .45em; padding-left: .45em;
  color: #f6f3e6; border-top: 1px solid #c9a227; border-bottom: 1px solid #c9a227; display: inline-block; line-height: 1.45; }

/* Pestañas */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #1f3a2f; padding: 12px 0 0 4px; background: transparent; }
.stTabs [data-baseweb="tab"] { height: 34px; padding: 0 20px; background: linear-gradient(180deg, #44644f, #2c4a3b);
  border: 1px solid #142a20; border-bottom: none; border-radius: 3px 3px 0 0; }
.stTabs [data-baseweb="tab"] p { font-size: 14.5px; color: #e9ecdf; font-family: 'IBM Plex Mono', monospace; }
.stTabs [aria-selected="true"] { background: #f1efe4 !important; box-shadow: inset 0 3px 0 #c9a227; }
.stTabs [aria-selected="true"] p { color: #1f2a24 !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 14px; }

/* Paneles */
div[class*="st-key-panel_"] { background: #eceadf; border: 1px solid #a9a796; border-radius: 2px;
  box-shadow: inset 0 1px 0 #fffdf4, 0 2px 6px rgba(0,0,0,.12); padding: 0 0 12px 0; gap: .55rem; }
div[class*="st-key-panel_"] > div { padding-left: 12px; padding-right: 12px; }
div[class*="st-key-panel_"] > div:first-child { padding: 0; }
.ph { background: linear-gradient(180deg, #305141, #243e32); color: #f3f1e6; font-family: 'Barlow Semi Condensed', sans-serif;
  font-weight: 600; font-size: 17px; letter-spacing: .06em; text-transform: uppercase; padding: 6px 12px;
  display: flex; justify-content: space-between; align-items: baseline; border-bottom: 2px solid #142a20; }
.ph small { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: #c9d3c4; text-transform: none; letter-spacing: .02em; }
div[class*="st-key-panel_"] label p { font-size: 10.5px !important; letter-spacing: .12em; text-transform: uppercase; color: #3d4a43; }

/* Tablas */
.mbt { width: 100%; border-collapse: collapse; font-size: 14px; color: #1f2a24; }
.mbt th { font-weight: 500; text-align: left; padding: 5px 8px; border-bottom: 1.5px solid #3c4a42; color: #2a362f; }
.mbt td { padding: 4px 8px; border-bottom: 1px solid #d0cdbe; }
.mbt .n { text-align: right; }
.mbt tr.hl td { background: #f3d97a; }
.mbt tr:hover td { background: #e2dfcf; }
.mbt tr.hl:hover td { background: #f0d264; }
.mbt img { height: 22px; vertical-align: middle; margin-right: 7px; }

.lo { font-size: 15px; }
.lo .r { display: flex; justify-content: space-between; padding: 6px 10px; border-bottom: 1px solid #d0cdbe; }
.lo .r.hl { background: #f3d97a; }
.bl { list-style: none; padding: 4px 2px 0 2px; margin: 0; font-size: 13.5px; line-height: 1.45; }
.bl li { display: flex; gap: 10px; margin: 8px 0; }
.bl li:before { content: ""; flex: 0 0 8px; height: 8px; margin-top: 6px; background: #4f7a63; }
.kv .r { display: flex; justify-content: space-between; padding: 4px 4px; border-bottom: 1px dotted #b9b6a6; font-size: 13.5px; }
.kv .r b { font-weight: 600; }

.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.tile { background: #f5f3ea; border: 1px solid #bdbaa9; padding: 8px 10px; }
.tile .k { font-size: 10.5px; letter-spacing: .14em; color: #52605a; }
.tile .v { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 700; font-size: 31px; color: #1e3a2f; line-height: 1.1; }
.tile .s { font-size: 11.5px; color: #5c6a63; min-height: 15px; }
.bar { height: 6px; background: #d6d3c4; margin-top: 6px; }
.bar > i { display: block; height: 100%; background: #2c4a3b; }
.bar > i.gold { background: #c9a227; }
.verdict { margin-top: 4px; padding: 8px; text-align: center; font-family: 'Barlow Semi Condensed', sans-serif;
  font-weight: 700; letter-spacing: .22em; font-size: 18px; }
.verdict.ok { background: #2c4a3b; color: #f3d97a; }
.verdict.no { background: #6f2f24; color: #f1efe4; }

.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip { padding: 3px 9px; border: 1px solid #a3aba4; font-size: 12.5px; background: #f5f3ea; color: #7a837d; }
.chip.on { background: #2c4a3b; color: #f3f1e6; border-color: #142a20; }

.card .r { display: grid; grid-template-columns: 26px 1fr 44px 56px 64px; padding: 5px 4px;
  border-bottom: 1px solid #d0cdbe; font-size: 14px; }
.card .r.h { color: #52605a; font-size: 11px; letter-spacing: .1em; border-bottom: 1.5px solid #3c4a42; }
.card .r span:nth-child(n+3) { text-align: right; }

/* Widgets */
.stButton button, .stDownloadButton button { background: #2c4a3b; color: #f1efe4; border: 1px solid #142a20;
  border-radius: 2px; letter-spacing: .06em; font-size: 13px; }
.stButton button:hover, .stDownloadButton button:hover { border-color: #c9a227; color: #f3d97a; background: #24402f; }
.stButton button p, .stDownloadButton button p { font-size: 13px; }

/* Scouting & Strike Zone (Movie Replica) */
.bio-card {
  display: flex; gap: 10px; background: #e5e2d4; border: 2px solid #2b3830;
  padding: 8px 10px; border-radius: 3px; box-shadow: inset 0 0 10px rgba(0,0,0,.08);
  box-sizing: border-box; width: 100%; max-width: 100%; overflow: hidden; align-items: stretch;
}
.bio-photo-wrap {
  position: relative; flex: 0 0 102px; width: 102px; height: 136px; max-height: 100%;
  background: #1a241f; border: 2px solid #142019; overflow: hidden !important;
  box-shadow: 0 2px 5px rgba(0,0,0,.3); box-sizing: border-box; border-radius: 2px;
}
.bio-photo {
  display: block; width: 100%; height: 100%; object-fit: cover; object-position: center top;
  filter: grayscale(100%) contrast(125%) brightness(95%);
}
.bio-photo-wrap:after {
  content: ""; position: absolute; top:0; left:0; width:100%; height:100%;
  background: repeating-linear-gradient(0deg, rgba(0,0,0,.15), rgba(0,0,0,.15) 1px, transparent 1px, transparent 2px);
  pointer-events: none;
}
.bio-table {
  width: 100%; table-layout: fixed; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 11.5px;
}
.bio-table td {
  padding: 2px 4px; border-bottom: 1px dotted #b0ac9d; line-height: 1.30;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.bio-table td.k {
  font-weight: 700; color: #1e3a2f; width: 38%;
}
.bio-table td.v {
  color: #141f19; font-weight: 500; width: 62%;
}

.mb-stat-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px 8px; font-family: 'IBM Plex Mono', monospace;
}
.mb-stat-item {
  display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: #1f2a24;
}
.mb-stat-item .k {
  font-weight: 700; color: #223e30; min-width: 28px;
}
.mb-stat-item .v {
  background: #dcdacd; border-top: 2px solid #858274; border-left: 2px solid #858274;
  border-right: 2px solid #faf8f2; border-bottom: 2px solid #faf8f2; padding: 1px 5px;
  min-width: 42px; text-align: right; font-weight: 600; font-size: 12.5px; color: #101a14;
}
img.mbw-img { vertical-align: middle; max-width: 100%; object-fit: contain; }

/* Footer */
.ftr { margin-top: 16px; background: linear-gradient(180deg, #27493b, #1b3a2e); border-top: 3px solid #10251c;
  padding: 10px 24px; display: flex; align-items: center; justify-content: space-between; border-radius: 0 0 2px 2px; }
.ftr .t { letter-spacing: .34em; color: #cfd6c6; font-size: 14px; }
.ftr .by { font-family: 'IBM Plex Mono', monospace; letter-spacing: .12em; color: #f3d97a; font-size: 13.5px; font-weight: 600; margin: 0 14px; display: inline-flex; align-items: center; }
.ftr img { height: 34px; }
</style>
"""
