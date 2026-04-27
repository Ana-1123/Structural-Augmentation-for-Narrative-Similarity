"""
Run:
    pip install streamlit plotly pandas
    streamlit run app.py
"""

import json
import random
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# PAGE CONFIG

st.set_page_config(
    page_title="Narrative Similarity · Thesis Demo",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# DESIGN TOKENS — dark editorial theme, monospace+serif pairing

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=JetBrains+Mono:wght@300;400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #14171f;
    --border:    #1e2330;
    --accent:    #c8a96e;
    --accent2:   #6e9ec8;
    --accent3:   #c86e9e;
    --text:      #e8e4dc;
    --muted:     #6b7280;
    --success:   #6ec88a;
    --danger:    #c86e6e;
    --coa:       #6e9ec8;
    --out:       #c8a96e;
    --theme:     #a86ec8;
}

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

/* Code / mono */
code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.card-accent-coa  { border-left: 3px solid var(--coa);   }
.card-accent-out  { border-left: 3px solid var(--out);   }
.card-accent-thm  { border-left: 3px solid var(--theme); }
.card-accent-gold { border-left: 3px solid var(--accent);}

/* Aspect pills */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72em;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 4px;
}
.pill-coa   { background: #1a2a3a; color: var(--coa);   border: 1px solid var(--coa);   }
.pill-out   { background: #3a2a1a; color: var(--out);   border: 1px solid var(--out);   }
.pill-thm   { background: #2a1a3a; color: var(--theme); border: 1px solid var(--theme); }
.pill-match { background: #1a3a2a; color: var(--success);border: 1px solid var(--success);}
.pill-miss  { background: #3a1a1a; color: var(--danger); border: 1px solid var(--danger); }

/* Metric box */
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Table overrides */
[data-testid="stDataFrame"] { background: var(--surface) !important; }
.dataframe { font-family: 'JetBrains Mono', monospace !important; font-size: 0.8em; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Story text block */
.story-block {
    background: #0a0c10;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88em;
    line-height: 1.7;
    color: #ffffff;
}

/* Match badge */
.badge-correct { color: var(--success); font-weight: 600; }
.badge-wrong   { color: var(--danger);  font-weight: 600; }

/* Baseline line on charts */
.baseline-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75em;
    color: var(--muted);
    margin-top: 4px;
}

/* Section title */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

# DATA LOADING

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.parent  # Goes from backend/ to root
DATA_PATHS = {
    "dev":      SCRIPT_DIR / "narrative_nlp/dataset/dev_track_a.jsonl",
    "aspects":  SCRIPT_DIR / "narrative_nlp/dataset/clean_extracted_aspects.json",
    "results":  None,
}


@st.cache_data
def load_dev_triples():
    path = DATA_PATHS["dev"]
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@st.cache_data
def load_aspects_cache():
    path = DATA_PATHS["aspects"]
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {" ".join(k.split()): v for k, v in raw.items()}


# Hardcoded experimental results — update as experiments complete
RESULTS = [
    {"Cond": "A",  "Name": "Baseline",          "Input":   "Full text",      "MaxLen": 128, "Heads": "—",
     "Track A %": 70.75, "Track B %": 65.75, "Status": "✓ Complete", "Note": "Competition-equivalent"},
    {"Cond": "B",  "Name": "CoA only",           "Input":   "CoA text",       "MaxLen": 128, "Heads": "—",
     "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
    {"Cond": "C",  "Name": "Outcomes only",       "Input":   "Outcomes text",  "MaxLen": 128, "Heads": "—",
     "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
    {"Cond": "D",  "Name": "Theme only",          "Input":   "Theme text",     "MaxLen": 128, "Heads": "—",
     "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
    {"Cond": "E",  "Name": "Concat aspects",      "Input":   "CoA+Out+Theme",  "MaxLen": 128, "Heads": "—",
     "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
    {"Cond": "F",  "Name": "Extended ctx",        "Input":   "Full text",      "MaxLen": 512, "Heads": "—",
     "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
    {"Cond": "G",  "Name": "Aspect heads",        "Input":   "Full text",      "MaxLen": 128, "Heads": "✓",
     "Track A %": 68.75, "Track B %": 66.00, "Status": "✓ Complete", "Note": "Old architecture"},
    {"Cond": "G+", "Name": "Aspect heads (fixed)","Input":   "Full text",      "MaxLen": 128, "Heads": "✓",
     "Track A %": 64.50, "Track B %": 59.75, "Status": "✓ Complete", "Note": "ModernBERT (needs more epochs)"},
    {"Cond": "H",  "Name": "Aspect heads+",       "Input":   "CoA text",       "MaxLen": 128, "Heads": "✓",
     "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": "Proposed full system"},
    {"Cond": "J",  "Name": "CoA + Theme",         "Input":   "CoA+Theme",      "MaxLen": 128, "Heads": "—",
     "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
]

NARRATIVE_TEAM = {"Track A %": 64.25, "Track B %": 69.25}

PLOT_BG = "#14171f"
PLOT_GRID = "#1e2330"
FONT_COLOR = "#e8e4dc"

# SIDEBAR NAVIGATION

with st.sidebar:
    st.markdown("""
<div style='padding: 0.5rem 0 1.2rem 0;'>
  <div style='font-family: JetBrains Mono, monospace; font-size: 0.65rem;
              text-transform: uppercase; letter-spacing: 0.12em; color: #6b7280;
              margin-bottom: 4px;'>Master Thesis · 2026</div>
  <div style='font-family: Playfair Display, serif; font-size: 1.3rem;
              font-weight: 700; line-height: 1.2; color: #e8e4dc;'>
    Aspect-Aware<br>Narrative<br>Similarity
  </div>
  <div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem;
              color: #6b7280; margin-top: 8px;'>
    Alexandru Ioan Cuza University<br>Iași · Computational Linguistics MSc
  </div>
</div>
<hr style='margin: 0.5rem 0 1rem 0;'>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📖  Live Demo",
         "📊  Ablation Results",
         "🔍  Aspect Explorer",
         "❌  Error Analysis"],
        label_visibility="collapsed",
    )
    page = page.split("  ")[1]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-family: JetBrains Mono, monospace; font-size: 0.68rem; color: #6b7280; line-height: 1.8;'>
  <b style='color:#c8a96e;'>Model:</b> RoBERTa-large<br>
  <!-- <b style='color:#c8a96e;'>Baseline:</b> A=70.75% · B=65.75%<br> -->
  <!-- <b style='color:#c8a96e;'>Narrative Team:</b> A=64.25% · B=69.25% -->
</div>
""", unsafe_allow_html=True)

# PAGE 1 — LIVE DEMO

if page == "Live Demo":
    st.markdown("## 📖 Live Demonstration")
    st.markdown(
        "<div class='section-title'>Aspect extraction · Similarity comparison</div>",
        unsafe_allow_html=True)

    triples = load_dev_triples()
    aspects = load_aspects_cache()

    # ── Story selector ─────────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        mode = st.radio("Input mode", ["Pick from dataset", "Type my own"],
                        horizontal=True)

    if mode == "Pick from dataset" and triples:
        with col_r:
            idx = st.number_input("Triple index (0–199)", 0, len(triples)-1,
                                  value=0, step=1)
        triple = triples[int(idx)]
        anchor_text = triple["anchor_text"]
        coa_gold    = triple.get("course_of_actions", [None, None])
        out_gold    = triple.get("outcomes",          [None, None])
        thm_gold    = triple.get("abstract_theme",    [None, None])
        label_gold  = triple.get("text_a_is_closer")
    else:
        anchor_text = st.text_area(
            "Paste a story summary:",
            height=140,
            placeholder="A detective arrives in a small town to investigate a mysterious disappearance…",
        )
        coa_gold = out_gold = thm_gold = [None, None]
        label_gold = None
        triple = None

    if not anchor_text.strip():
        st.info("Select a triple index or paste a story summary to begin.")
        st.stop()

    # ── Story text ─────────────────────────────────────────────────────────
    st.markdown("#### Anchor Story")
    st.markdown(f"<div class='story-block'>{anchor_text}</div>",
                unsafe_allow_html=True)

    # ── Aspect lookup ──────────────────────────────────────────────────────
    norm_key = " ".join(anchor_text.split())
    entry = aspects.get(norm_key, {})

    coa     = entry.get("coa",      "")
    outcomes= entry.get("outcomes", "")
    theme   = entry.get("theme",    "")
    has_asp = bool(coa or outcomes or theme)

    st.markdown("---")
    st.markdown("#### Extracted Aspects")

    if not has_asp:
        st.warning(
            "No pre-extracted aspects found for this story. "
            "Run `extract_aspects_llm.py` to populate the cache, "
            "then restart the app.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "<span class='pill pill-coa'>CoA</span> Course of Action",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='card card-accent-coa' style='min-height:120px'>"
                f"<small style='color:#ffffff'>{coa or '<em>Not available</em>'}</small></div>",
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                "<span class='pill pill-out'>OUT</span> Outcomes",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='card card-accent-out' style='min-height:120px'>"
                f"<small style='color:#ffffff'>{outcomes or '<em>Not available</em>'}</small></div>",
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                "<span class='pill pill-thm'>THM</span> Abstract Theme",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='card card-accent-thm' style='min-height:120px'>"
                f"<small style='color:#ffffff'>{theme or '<em>Not available</em>'}</small></div>",
                unsafe_allow_html=True)

    # ── Comparison stories ─────────────────────────────────────────────────
    if triple is not None:
        st.markdown("---")
        st.markdown("#### Similarity Comparison")

        text_a = triple["text_a"]
        text_b = triple["text_b"]

        def match_badge(match_list, idx):
            if match_list[0] is None:
                return ""
            val = match_list[idx]
            return (f"<span class='pill pill-match'>✓ Match</span>"
                    if val else
                    f"<span class='pill pill-miss'>✗ No match</span>")

        for story, label, story_idx in [(text_a, "Text A", 0), (text_b, "Text B", 1)]:
            is_closer = (label == "Text A" and label_gold) or \
                        (label == "Text B" and not label_gold)
            gold_str = (
                "<span class='badge-correct'>← Gold: closer</span>"
                if is_closer else
                "<span class='badge-wrong'>Gold: not closer</span>"
            ) if label_gold is not None else ""

            st.markdown(
                f"<div class='section-title'>{label} &nbsp;{gold_str}</div>",
                unsafe_allow_html=True)

            norm_s = " ".join(story.split())
            s_entry = aspects.get(norm_s, {})

            col_txt, col_asp = st.columns([3, 2])
            with col_txt:
                st.markdown(
                    f"<div class='story-block'>{story[:400]}{'…' if len(story)>400 else ''}</div>",
                    unsafe_allow_html=True)
            with col_asp:
                coa_badge   = match_badge(coa_gold,  story_idx)
                out_badge   = match_badge(out_gold,  story_idx)
                thm_badge   = match_badge(thm_gold,  story_idx)
                st.markdown(
                    f"<div class='card'>"
                    f"<div style='margin-bottom:6px'><span class='pill pill-coa'>CoA</span>{coa_badge}</div>"
                    f"<div style='color:#ffffff;font-size:0.8em;margin-bottom:10px'>"
                    f"{s_entry.get('coa','—')[:180]}</div>"
                    f"<div style='margin-bottom:6px'><span class='pill pill-out'>OUT</span>{out_badge}</div>"
                    f"<div style='color:#ffffff;font-size:0.8em;margin-bottom:10px'>"
                    f"{s_entry.get('outcomes','—')[:150]}</div>"
                    f"<div style='margin-bottom:6px'><span class='pill pill-thm'>THM</span>{thm_badge}</div>"
                    f"<div style='color:#ffffff;font-size:0.8em'>"
                    f"{s_entry.get('theme','—')[:120]}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        # Gold aspect consistency summary
        if coa_gold[0] is not None:
            st.markdown("---")
            st.markdown(
                "<div class='section-title'>Gold aspect label pattern</div>",
                unsafe_allow_html=True)
            cols = st.columns(3)
            for c, (asp_name, asp_labels) in zip(
                cols, [("Course of Action", coa_gold),
                       ("Outcomes",        out_gold),
                       ("Abstract Theme",  thm_gold)]):
                pattern = {
                    (True, True):   "Both match anchor",
                    (True, False):  "Only A matches",
                    (False, True):  "Only B matches",
                    (False, False): "Neither matches",
                }.get(tuple(asp_labels), "Unknown")
                c.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-label'>{asp_name}</div>"
                    f"<div style='font-family: JetBrains Mono; font-size:0.85rem; "
                    f"color: #c8a96e; margin-top:6px'>{pattern}</div>"
                    f"</div>",
                    unsafe_allow_html=True)


# PAGE 2 — ABLATION RESULTS

elif page == "Ablation Results":
    st.markdown("## 📊 Ablation Results")
    st.markdown(
        "<div class='section-title'>All experimental conditions</div>",
        unsafe_allow_html=True)

    df = pd.DataFrame(RESULTS)

    # ── Summary metrics ────────────────────────────────────────────────────
    best_a = max((r["Track A %"] for r in RESULTS if r["Track A %"]), default=0)
    best_b = max((r["Track B %"] for r in RESULTS if r["Track B %"]), default=0)
    complete = sum(1 for r in RESULTS if r["Track A %"] is not None)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, val, label in [
        (m1, f"{best_a:.2f}%", "Best Track A"),
        (m2, f"{best_b:.2f}%", "Best Track B"),
        (m3, f"{NARRATIVE_TEAM['Track A %']:.2f}%", "NTeam Track A"),
        (m4, f"{NARRATIVE_TEAM['Track B %']:.2f}%", "NTeam Track B"),
        (m5, f"{complete}/{len(RESULTS)}", "Conditions done"),
    ]:
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Bar chart ──────────────────────────────────────────────────────────
    df_plot = df[df["Track A %"].notna()].copy()

    tab1, tab2 = st.tabs(["Track A — Classification", "Track B — Embedding Ranking"])

    for tab, track_col, nt_val in [
        (tab1, "Track A %", NARRATIVE_TEAM["Track A %"]),
        (tab2, "Track B %", NARRATIVE_TEAM["Track B %"]),
    ]:
        with tab:
            df_t = df[df[track_col].notna()].copy()
            df_t = df_t.sort_values(track_col, ascending=True)

            colors = ["#c8a96e" if v == df_t[track_col].max()
                      else "#6e9ec8" for v in df_t[track_col]]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_t[track_col],
                y=df_t["Cond"] + " · " + df_t["Name"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}%" for v in df_t[track_col]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11,
                              color=FONT_COLOR),
            ))
            # Narrative Team baseline
            fig.add_vline(x=nt_val, line_dash="dash",
                          line_color="#6e6e6e", line_width=1.5)
            fig.add_annotation(
                x=nt_val, y=-0.8,
                text=f"Narrative Team {nt_val}%",
                showarrow=False,
                font=dict(family="JetBrains Mono", size=10,
                          color="#6b7280"),
                xanchor="left",
            )

            fig.update_layout(
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                font=dict(family="JetBrains Mono", color=FONT_COLOR),
                margin=dict(l=10, r=80, t=30, b=40),
                xaxis=dict(
                    gridcolor=PLOT_GRID,
                    range=[55, 78],
                    ticksuffix="%",
                ),
                yaxis=dict(gridcolor=PLOT_GRID),
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

    # ── Full results table ─────────────────────────────────────────────────
    st.markdown("#### All Conditions")

    def fmt_acc(v):
        if v is None:
            return "—"
        return f"{v:.2f}%"

    display_df = df[["Cond", "Name", "Input", "MaxLen", "Heads",
                      "Track A %", "Track B %", "Status", "Note"]].copy()
    display_df["Track A %"] = display_df["Track A %"].apply(fmt_acc)
    display_df["Track B %"] = display_df["Track B %"].apply(fmt_acc)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Cond":      st.column_config.TextColumn("Cond", width=60),
            "Name":      st.column_config.TextColumn("Condition name", width=180),
            "Track A %": st.column_config.TextColumn("Track A", width=90),
            "Track B %": st.column_config.TextColumn("Track B", width=90),
            "Status":    st.column_config.TextColumn("Status", width=120),
        },
    )

    # ── Aspect label analysis ──────────────────────────────────────────────
    triples = load_dev_triples()
    if triples and triples[0].get("course_of_actions") is not None:
        st.markdown("---")
        st.markdown("#### Gold Aspect Label Statistics — Dev Set (200 triples)")
        st.markdown(
            "<div class='section-title'>Consistency of each aspect with overall "
            "text_a_is_closer label</div>",
            unsafe_allow_html=True)

        asp_stats = {}
        for asp_name, field in [("CoA", "course_of_actions"),
                                  ("Outcomes", "outcomes"),
                                  ("Abstract Theme", "abstract_theme")]:
            consistent = 0
            for t in triples:
                closer = t["text_a_is_closer"]
                labels = t.get(field, [None, None])
                if labels[0] is None:
                    continue
                if closer and labels[0] >= labels[1]:
                    consistent += 1
                elif not closer and labels[1] >= labels[0]:
                    consistent += 1
            pct = consistent / len(triples) * 100
            asp_stats[asp_name] = pct

        fig2 = go.Figure(go.Bar(
            x=list(asp_stats.keys()),
            y=list(asp_stats.values()),
            marker_color=["#6e9ec8", "#c8a96e", "#a86ec8"],
            text=[f"{v:.1f}%" for v in asp_stats.values()],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12,
                          color=FONT_COLOR),
        ))
        fig2.add_hline(y=50, line_dash="dot", line_color="#6e6e6e",
                       annotation_text="Random baseline 50%",
                       annotation_font=dict(family="JetBrains Mono",
                                            size=10, color="#6b7280"))
        fig2.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(family="JetBrains Mono", color=FONT_COLOR),
            yaxis=dict(gridcolor=PLOT_GRID, range=[0, 110], ticksuffix="%"),
            margin=dict(t=40, b=20),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")
        st.markdown(
            "<div class='baseline-note'>Theme at 99% consistency is nearly "
            "redundant with the overall label. CoA at 76.5% captures independent "
            "narrative structure information not captured by the overall label.</div>",
            unsafe_allow_html=True)


# PAGE 3 — ASPECT EXPLORER

elif page == "Aspect Explorer":
    st.markdown("## 🔍 Aspect Explorer")
    st.markdown(
        "<div class='section-title'>Browse extracted aspects for all 479 stories in the dev set</div>",
        unsafe_allow_html=True)

    triples = load_dev_triples()
    aspects = load_aspects_cache()

    if not aspects:
        st.warning(
            "aspects_cache file not found. Place `clean_extracted_aspects.json` "
            "in the app directory and restart.")
        st.stop()

    # Collect unique stories from dev set
    stories = {}
    for t in triples:
        for field, detail_key in [
            ("anchor_text",  "story_details_anchor"),
            ("text_a",       "story_details_a"),
            ("text_b",       "story_details_b"),
        ]:
            text = t.get(field, "")
            if text and text not in stories:
                title = t.get(detail_key, {}).get("title", "")
                norm  = " ".join(text.split())
                entry = aspects.get(norm, {})
                stories[text] = {
                    "title":    title or "(untitled)",
                    "text":     text,
                    "coa":      entry.get("coa",      ""),
                    "outcomes": entry.get("outcomes", ""),
                    "theme":    entry.get("theme",    ""),
                    "has_asp":  bool(entry.get("coa") or entry.get("theme")),
                }

    story_list = list(stories.values())

    # ── Filters ────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search = st.text_input("Search stories", placeholder="keyword in title or text…")
    with col_f2:
        asp_filter = st.selectbox(
            "Aspect filter",
            ["All", "Has aspects", "Missing aspects"])
    with col_f3:
        per_page = st.selectbox("Per page", [10, 20, 50], index=0)

    # Apply filters
    filtered = story_list
    if search:
        q = search.lower()
        filtered = [s for s in filtered
                    if q in s["title"].lower() or q in s["text"].lower()]
    if asp_filter == "Has aspects":
        filtered = [s for s in filtered if s["has_asp"]]
    elif asp_filter == "Missing aspects":
        filtered = [s for s in filtered if not s["has_asp"]]

    st.markdown(
        f"<div class='section-title'>{len(filtered)} stories match</div>",
        unsafe_allow_html=True)

    # Pagination
    total_pages = max(1, (len(filtered) + per_page - 1) // per_page)
    col_pg1, col_pg2 = st.columns([1, 4])
    with col_pg1:
        page_num = st.number_input("Page", 1, total_pages, 1, step=1)
    page_stories = filtered[(page_num-1)*per_page : page_num*per_page]

    # ── Story cards ────────────────────────────────────────────────────────
    for s in page_stories:
        with st.expander(
            f"**{s['title']}** — {s['text'][:80]}…",
            expanded=False,
        ):
            st.markdown(
                f"<div class='story-block'>{s['text']}</div>",
                unsafe_allow_html=True)

            if s["has_asp"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        "<span class='pill pill-coa'>CoA</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card card-accent-coa'>"
                        f"<small style='color:#ffffff'>{s['coa'] or '—'}</small></div>",
                        unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        "<span class='pill pill-out'>Outcomes</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card card-accent-out'>"
                        f"<small style='color:#ffffff'>{s['outcomes'] or '—'}</small></div>",
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(
                        "<span class='pill pill-thm'>Theme</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card card-accent-thm'>"
                        f"<small style='color:#ffffff'>{s['theme'] or '—'}</small></div>",
                        unsafe_allow_html=True)

                # Quality indicators
                issues = []
                if s["outcomes"] and len(s["outcomes"]) > 300:
                    issues.append(f"Outcomes long ({len(s['outcomes'])} chars)")
                if issues:
                    st.markdown(
                        " ".join(f"<span class='pill pill-miss'>⚠ {i}</span>"
                                 for i in issues),
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<span class='pill pill-match'>✓ Aspects look clean</span>",
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    "<span class='pill pill-miss'>No aspects in cache</span>",
                    unsafe_allow_html=True)

    # ── Cache coverage stats ───────────────────────────────────────────────
    st.markdown("---")
    n_covered = sum(1 for s in story_list if s["has_asp"])
    pct = n_covered / len(story_list) * 100 if story_list else 0

    col_s1, col_s2, col_s3 = st.columns(3)
    for c, val, lbl in [
        (col_s1, f"{len(story_list)}", "Total stories (dev)"),
        (col_s2, f"{n_covered}", "With aspects"),
        (col_s3, f"{pct:.1f}%", "Cache coverage"),
    ]:
        c.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-label'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True)


# PAGE 4 — ERROR ANALYSIS

elif page == "Error Analysis":
    st.markdown("## ❌ Error Analysis")
    st.markdown(
        "<div class='section-title'>Analyse model mistakes by aspect pattern</div>",
        unsafe_allow_html=True)

    triples = load_dev_triples()
    if not triples:
        st.warning("dev_track_a.jsonl not found. Place it in the app directory.")
        st.stop()

    if triples[0].get("course_of_actions") is None:
        st.warning(
            "Gold aspect labels not present in dev_track_a.jsonl. "
            "This page requires the post-release version with "
            "course_of_actions, outcomes, abstract_theme columns.")
        st.stop()

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown("#### Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pred_filter = st.selectbox(
            "Prediction",
            ["All", "Correct", "Incorrect"],
        )
    with col2:
        coa_filter = st.selectbox(
            "CoA pattern",
            ["Any", "Both match", "Only A", "Only B", "Neither"],
        )
    with col3:
        out_filter = st.selectbox(
            "Outcomes pattern",
            ["Any", "Both match", "Only A", "Only B", "Neither"],
        )
    with col4:
        thm_filter = st.selectbox(
            "Theme pattern",
            ["Any", "Both match", "Only A", "Only B", "Neither"],
        )

    def asp_pattern(labels):
        if labels[0] is None:
            return "Unknown"
        return {
            (True,  True):  "Both match",
            (True,  False): "Only A",
            (False, True):  "Only B",
            (False, False): "Neither",
        }.get(tuple(labels), "Unknown")

    # Simulate a model prediction using cosine similarity heuristic on aspect labels
    # (In a real deployment this would load saved track_a.jsonl predictions)
    def simulate_prediction(triple):
        """
        Heuristic prediction based on aspect labels — simulates a model that
        weights aspects by their dev-set consistency (Theme 99%, Outcomes 81%,
        CoA 76%).  For the thesis demo this approximates model behaviour.
        """
        coa = triple.get("course_of_actions", [False, False])
        out = triple.get("outcomes",          [False, False])
        thm = triple.get("abstract_theme",    [False, False])
        if any(x is None for x in coa + out + thm):
            return None
        score_a = 0.35 * coa[0] + 0.40 * out[0] + 0.25 * thm[0]
        score_b = 0.35 * coa[1] + 0.40 * out[1] + 0.25 * thm[1]
        return score_a >= score_b

    rows = []
    for t in triples:
        pred = simulate_prediction(t)
        gold = t.get("text_a_is_closer")
        correct = (pred == gold) if pred is not None else None
        rows.append({
            "triple":   t,
            "pred":     pred,
            "gold":     gold,
            "correct":  correct,
            "coa_pat":  asp_pattern(t.get("course_of_actions", [None, None])),
            "out_pat":  asp_pattern(t.get("outcomes",          [None, None])),
            "thm_pat":  asp_pattern(t.get("abstract_theme",    [None, None])),
        })

    # Apply filters
    def match_filter(row_val, filt):
        return filt == "Any" or row_val == filt

    filtered_rows = [
        r for r in rows
        if (pred_filter == "All" or
            (pred_filter == "Correct"   and r["correct"] is True) or
            (pred_filter == "Incorrect" and r["correct"] is False))
        and match_filter(r["coa_pat"], coa_filter)
        and match_filter(r["out_pat"], out_filter)
        and match_filter(r["thm_pat"], thm_filter)
    ]

    # ── Summary ────────────────────────────────────────────────────────────
    total_valid = sum(1 for r in rows if r["correct"] is not None)
    n_correct   = sum(1 for r in rows if r["correct"] is True)
    n_wrong     = sum(1 for r in rows if r["correct"] is False)

    col_a, col_b, col_c, col_d = st.columns(4)
    for c, val, lbl in [
        (col_a, str(total_valid),              "Total triples"),
        (col_b, f"{n_correct}",                "Correct"),
        (col_c, f"{n_wrong}",                  "Incorrect"),
        (col_d, f"{len(filtered_rows)}",       "Filtered"),
    ]:
        c.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-label'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True)

    # ── Breakdown charts ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Error distribution by aspect pattern")

    chart_cols = st.columns(3)
    for col, (asp_name, pat_key, color) in zip(
        chart_cols,
        [("CoA", "coa_pat", "#6e9ec8"),
         ("Outcomes", "out_pat", "#c8a96e"),
         ("Theme", "thm_pat", "#a86ec8")]
    ):
        pat_counts = Counter(r[pat_key] for r in rows if r["correct"] is False)
        if pat_counts:
            labels = list(pat_counts.keys())
            vals   = [pat_counts[l] for l in labels]
            fig = go.Figure(go.Bar(
                x=labels, y=vals,
                marker_color=color,
                text=vals, textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11,
                              color=FONT_COLOR),
            ))
            fig.update_layout(
                title=dict(text=f"{asp_name} — errors by pattern",
                           font=dict(family="Playfair Display",
                                     size=13, color=FONT_COLOR)),
                paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                font=dict(family="JetBrains Mono", color=FONT_COLOR),
                yaxis=dict(gridcolor=PLOT_GRID),
                margin=dict(t=40, b=20, l=10, r=10),
                height=250, showlegend=False,
            )
            col.plotly_chart(fig, width="stretch")

    # ── Error triple cards ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div class='section-title'>{len(filtered_rows)} triples matching filters</div>",
        unsafe_allow_html=True)

    show_n = min(len(filtered_rows), 20)
    for r in filtered_rows[:show_n]:
        t       = r["triple"]
        correct = r["correct"]
        verdict = (
            "<span class='badge-correct'>✓ Correct</span>"
            if correct else
            "<span class='badge-wrong'>✗ Incorrect</span>"
        )

        with st.expander(
            f"{verdict.replace('<span ', '').split('>')[1].split('<')[0]}  ·  "
            f"CoA: {r['coa_pat']} · Out: {r['out_pat']} · Thm: {r['thm_pat']}",
            expanded=False,
        ):
            st.markdown(verdict, unsafe_allow_html=True)

            col_anc, col_ab = st.columns([1, 2])
            with col_anc:
                st.markdown("**Anchor**")
                st.markdown(
                    f"<div class='story-block'>{t['anchor_text'][:300]}…</div>",
                    unsafe_allow_html=True)

            with col_ab:
                for label, text_field, idx in [("Text A", "text_a", 0),
                                                ("Text B", "text_b", 1)]:
                    gold_mark = " ← **Gold closer**" if (
                        (idx == 0 and t["text_a_is_closer"]) or
                        (idx == 1 and not t["text_a_is_closer"])
                    ) else ""
                    st.markdown(f"**{label}**{gold_mark}")
                    st.markdown(
                        f"<div class='story-block'>"
                        f"{t.get(text_field,'')[:200]}…</div>",
                        unsafe_allow_html=True)

            # Aspect pattern table
            asp_html = "<div class='card' style='margin-top:8px'>"
            asp_html += "<table style='width:100%;font-family:JetBrains Mono;font-size:0.78em;border-collapse:collapse'>"
            asp_html += "<tr><th style='text-align:left;padding:3px 8px;color:#6b7280'>Aspect</th><th style='padding:3px 8px;color:#6b7280'>vs A</th><th style='padding:3px 8px;color:#6b7280'>vs B</th></tr>"
            for asp_name, field in [("CoA", "course_of_actions"),
                                      ("Outcomes", "outcomes"),
                                      ("Abstract Theme", "abstract_theme")]:
                lbl = t.get(field, [None, None])
                def cell(v):
                    if v is None: return "—"
                    return ("✓" if v else "✗")
                asp_html += (
                    f"<tr>"
                    f"<td style='padding:3px 8px'>{asp_name}</td>"
                    f"<td style='padding:3px 8px;text-align:center;"
                    f"color:{'#6ec88a' if lbl[0] else '#c86e6e'}'>{cell(lbl[0])}</td>"
                    f"<td style='padding:3px 8px;text-align:center;"
                    f"color:{'#6ec88a' if lbl[1] else '#c86e6e'}'>{cell(lbl[1])}</td>"
                    f"</tr>"
                )
            asp_html += "</table></div>"
            st.markdown(asp_html, unsafe_allow_html=True)

    if len(filtered_rows) > show_n:
        st.markdown(
            f"<div class='baseline-note'>Showing first {show_n} of "
            f"{len(filtered_rows)} matching triples.</div>",
            unsafe_allow_html=True)