"""
TRAXR — Track. Prove. Advance.
Premium Streamlit app: Landing page → Dashboard with 360° readiness analysis.
"""
from __future__ import annotations
import hashlib
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import demodata
from nlppipeline import extract_pdf_text, extract_profile_signals
from scorer import compute_360_score
from llmclient import (
    extract_role_dna, generate_roadmap, evaluate_skillprint,
    generate_skillprint_challenge, is_llm_available,
)

st.set_page_config(page_title="TRAXR – Track. Prove. Advance.", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ═══════════════════════════════════════════════════════════
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*{font-family:'Inter',system-ui,sans-serif}

/* ── Tokens ─────────────────────────── */
:root{
--bg:#080b14;--bg2:#0e1525;--card:#141c30;--card-h:#192240;--elev:#1e2a45;
--bdr:rgba(148,163,184,.08);--bdr-h:rgba(148,163,184,.16);
--acc:#6366f1;--acc2:#818cf8;--acc3:#a5b4fc;--glow:rgba(99,102,241,.12);
--t1:#f1f5f9;--t2:#94a3b8;--t3:#64748b;
--ok:#34d399;--warn:#fbbf24;--err:#f87171;
--r:14px;--r2:10px;
--sh:0 4px 24px rgba(0,0,0,.3);--sh-h:0 12px 40px rgba(0,0,0,.4);
}

/* ── Global ─────────────────────────── */
.stApp{background:var(--bg)!important;color:var(--t1)}
.stApp::before{content:'';position:fixed;inset:0;
  background:radial-gradient(800px circle at 15% 15%,rgba(99,102,241,.06),transparent 60%),
  radial-gradient(600px circle at 85% 60%,rgba(139,92,246,.05),transparent 55%),
  radial-gradient(500px circle at 50% 90%,rgba(79,70,229,.04),transparent 50%);
  pointer-events:none;z-index:0}
.block-container{max-width:1200px!important;padding-top:1.5rem!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{display:none}
hr{border-color:var(--bdr)!important}

/* ── Animations ─────────────────────── */
@keyframes shimmer{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes fadeUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes fillBar{from{width:0}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 15px var(--glow)}50%{box-shadow:0 0 35px rgba(99,102,241,.25)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes spinSlow{from{transform:rotate(0)}to{transform:rotate(360deg)}}
@keyframes typewriter{from{width:0}to{width:100%}}
@keyframes blink{50%{border-color:transparent}}
@keyframes slideR{from{opacity:0;transform:translateX(-30px)}to{opacity:1;transform:translateX(0)}}

/* ── Logo Shimmer ───────────────────── */
.logo-shimmer{
  font-size:4.5rem;font-weight:900;letter-spacing:-3px;line-height:1;
  background:linear-gradient(120deg,#6366f1 0%,#a78bfa 20%,#f1f5f9 45%,#a78bfa 65%,#6366f1 100%);
  background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shimmer 4s linear infinite;display:inline-block}
.logo-sm{font-size:1.6rem;letter-spacing:-1.5px;font-weight:800;
  background:linear-gradient(120deg,#6366f1,#a78bfa,#6366f1);background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 4s linear infinite}

/* ── Landing ────────────────────────── */
.land-hero{text-align:center;padding:4rem 0 2rem;animation:fadeIn .8s ease}
.land-tagline{font-size:1.3rem;font-weight:600;color:var(--acc3);letter-spacing:3px;margin:.8rem 0 .4rem;
  animation:fadeUp .7s ease .2s both}
.land-sub{color:var(--t2);font-size:1.05rem;max-width:640px;margin:.6rem auto 2rem;line-height:1.6;
  animation:fadeUp .7s ease .4s both}
.land-section{animation:fadeUp .6s ease both}
.land-section h2{font-size:1.7rem;font-weight:800;text-align:center;margin-bottom:.3rem;color:var(--t1)}
.land-section .sec-sub{text-align:center;color:var(--t2);margin-bottom:2rem;font-size:.95rem}

.feat-card{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);padding:1.8rem 1.5rem;
  text-align:center;transition:all .3s ease;animation:fadeUp .6s ease both;min-height:220px}
.feat-card:hover{transform:translateY(-6px);box-shadow:var(--sh-h);border-color:var(--bdr-h);background:var(--card-h)}
.feat-card .fi{font-size:2.2rem;margin-bottom:.8rem;display:block}
.feat-card h3{font-size:1.05rem;font-weight:700;margin-bottom:.5rem;color:var(--t1)}
.feat-card p{font-size:.88rem;color:var(--t2);line-height:1.55}

.step-card{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);padding:1.5rem;
  text-align:center;transition:all .3s ease;animation:fadeUp .6s ease both;position:relative}
.step-card:hover{border-color:var(--acc);transform:translateY(-4px)}
.step-num{width:36px;height:36px;border-radius:50%;background:var(--acc);color:#fff;font-weight:800;
  font-size:.95rem;display:inline-flex;align-items:center;justify-content:center;margin-bottom:.7rem}
.step-card h4{font-size:.95rem;font-weight:700;color:var(--t1);margin-bottom:.3rem}
.step-card p{font-size:.82rem;color:var(--t2);line-height:1.5}

.dim-card-sm{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r2);padding:1rem;
  text-align:center;transition:all .25s ease;animation:fadeUp .5s ease both}
.dim-card-sm:hover{border-color:var(--acc);background:var(--card-h)}
.dim-card-sm .di{font-size:1.5rem;margin-bottom:.4rem}
.dim-card-sm h4{font-size:.82rem;font-weight:600;color:var(--t1);margin:0}

.preview-card{background:var(--card);border:1px solid var(--bdr);border-radius:18px;padding:1.5rem;
  animation:fadeUp .8s ease .5s both;box-shadow:var(--sh)}

/* ── Dashboard ──────────────────────── */
.dash-hdr{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0 1rem;
  border-bottom:1px solid var(--bdr);margin-bottom:1.2rem;animation:fadeIn .5s ease}
.dash-hdr-left{display:flex;align-items:center;gap:1rem}
.dash-greeting{font-size:1.15rem;font-weight:600;color:var(--t1);overflow:hidden;white-space:nowrap;
  border-right:2px solid var(--acc);animation:typewriter 1.5s steps(30) both,blink .8s step-end infinite;
  display:inline-block;max-width:100%}

.d-card{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);padding:1.3rem 1.5rem;
  transition:all .3s ease;animation:fadeUp .5s ease both}
.d-card:hover{border-color:var(--bdr-h);box-shadow:var(--sh);background:var(--card-h)}
.d-card-title{font-size:.92rem;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:.8rem}

/* ── Score Circle ───────────────────── */
.score-circle{width:170px;height:170px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  margin:0 auto;position:relative;animation:fadeIn .8s ease .2s both}
.score-inner{width:140px;height:140px;border-radius:50%;background:var(--card);display:flex;flex-direction:column;
  align-items:center;justify-content:center;position:relative;z-index:1}
.score-val{font-size:3.2rem;font-weight:900;line-height:1}
.score-label{font-size:.78rem;color:var(--t2);margin-top:.2rem;font-weight:500}

/* ── Progress Bars ──────────────────── */
.pbar-wrap{margin:.55rem 0}
.pbar-head{display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:4px}
.pbar-name{color:var(--t2);font-weight:500}
.pbar-val{font-weight:700}
.pbar-track{background:rgba(30,42,69,.8);border-radius:6px;height:8px;overflow:hidden}
.pbar-fill{height:100%;border-radius:6px;animation:fillBar 1.2s ease both}

/* ── Chips ──────────────────────────── */
.chip{display:inline-block;padding:4px 12px;margin:3px;border-radius:20px;font-size:.78rem;font-weight:600;
  color:#fff;transition:all .2s ease;cursor:default}
.chip:hover{transform:translateY(-2px);filter:brightness(1.15)}
.chip-hard{background:var(--err)}.chip-soft{background:var(--warn);color:#1a1a1a}
.chip-context{background:#6366f1}.chip-matched{background:var(--ok);color:#1a1a1a}

/* ── Roadmap ────────────────────────── */
.road-block{background:var(--card);border-left:3px solid var(--acc);border-radius:0 var(--r2) var(--r2) 0;
  padding:1rem 1.2rem;margin-bottom:.7rem;transition:all .2s ease}
.road-block:hover{border-left-color:var(--acc2);background:var(--card-h)}
.road-block h4{color:var(--acc2);margin:0 0 .4rem;font-size:.92rem;font-weight:700}
.road-block ul{margin:0;padding-left:1.2rem;color:var(--t2);font-size:.85rem}
.road-block a{color:var(--acc2)}

/* ── Component Overrides ────────────── */
.stTextArea textarea{background:var(--bg2)!important;color:var(--t1)!important;border:1px solid var(--bdr)!important;
  border-radius:var(--r2)!important;transition:border-color .2s!important}
.stTextArea textarea:focus{border-color:var(--acc)!important}
div[data-testid="stFileUploader"]{background:var(--bg2);border:1px dashed rgba(99,102,241,.25);
  border-radius:var(--r2);padding:.8rem;transition:border-color .3s}
div[data-testid="stFileUploader"]:hover{border-color:var(--acc)}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#6366f1,#818cf8)!important;color:#fff!important;border:none!important;
  border-radius:var(--r2)!important;padding:.6rem 1.8rem!important;font-weight:700!important;font-size:.9rem!important;
  transition:all .25s ease!important;letter-spacing:.3px!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(99,102,241,.4)!important}
.stButton>button:active{transform:translateY(0)!important}

/* CTA special */
.cta-primary .stButton>button{font-size:1.05rem!important;padding:.75rem 2.5rem!important;border-radius:12px!important;
  animation:pulseGlow 3s ease-in-out infinite}
.cta-ghost .stButton>button{background:transparent!important;border:1.5px solid var(--bdr-h)!important;color:var(--t2)!important}
.cta-ghost .stButton>button:hover{border-color:var(--acc)!important;color:var(--t1)!important;background:var(--glow)!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--bdr)}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--t3);border:none;border-bottom:2px solid transparent;
  border-radius:0;padding:.6rem 1rem;font-weight:600;font-size:.82rem;transition:all .2s}
.stTabs [data-baseweb="tab"]:hover{color:var(--t1)}
.stTabs [aria-selected="true"]{color:var(--acc2)!important;border-bottom-color:var(--acc)!important;background:transparent!important}

/* Expander */
.streamlit-expanderHeader{background:var(--card)!important;border-radius:var(--r2)!important;font-weight:600!important;
  border:1px solid var(--bdr)!important;transition:all .2s!important}
.streamlit-expanderHeader:hover{border-color:var(--bdr-h)!important;background:var(--card-h)!important}

/* Metrics */
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r2);
  padding:.8rem 1rem;text-align:center}
[data-testid="stMetricLabel"]{color:var(--t3)!important;font-weight:500!important}
[data-testid="stMetricValue"]{color:var(--t1)!important;font-weight:800!important}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
_DEFAULTS = {
    "view": "landing", "raw_text": "", "jd_text": "", "input_hash": "",
    "resume_source": None, "role_dna": None, "profile_signals": None,
    "scores": None, "gaps": None, "roadmap": None,
    "skillprint_challenge": None, "skillprint_code": "",
    "skillprint_result": None, "analysis_complete": False, "using_demo": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def _h(t): return hashlib.md5(t.encode()).hexdigest()

def _sc(s):
    if s >= 75: return "var(--ok)"
    if s >= 50: return "var(--warn)"
    return "var(--err)"

def _sc_hex(s):
    if s >= 75: return "#34d399"
    if s >= 50: return "#fbbf24"
    return "#f87171"

def build_radar(dims, h=380, fill_color="rgba(99,102,241,0.12)", line_color="#818cf8"):
    cats = list(dims.keys()); vals = list(dims.values())
    cats += [cats[0]]; vals += [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
        fillcolor=fill_color, line=dict(color=line_color, width=2.5),
        marker=dict(size=5, color=line_color)))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(color="#64748b",size=9),
                gridcolor="rgba(148,163,184,.08)", linecolor="rgba(148,163,184,.05)"),
            angularaxis=dict(tickfont=dict(color="#94a3b8",size=10),
                gridcolor="rgba(148,163,184,.08)", linecolor="rgba(148,163,184,.05)")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(l=70,r=70,t=30,b=30), height=h)
    return fig

def pbar(label, score, delay=0):
    c = _sc_hex(score)
    st.markdown(f"""<div class="pbar-wrap"><div class="pbar-head">
        <span class="pbar-name">{label}</span><span class="pbar-val" style="color:{c}">{score}</span>
    </div><div class="pbar-track"><div class="pbar-fill" style="width:{score}%;background:{c};animation-delay:{delay}s"></div></div></div>""", unsafe_allow_html=True)

def score_circle(score):
    c = _sc_hex(score)
    st.markdown(f"""<div class="score-circle" style="background:conic-gradient({c} {score*3.6}deg, rgba(30,42,69,.8) 0deg)">
        <div class="score-inner"><span class="score-val" style="color:{c}">{score}</span>
        <span class="score-label">Readiness Score</span></div></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════
def render_landing():
    # ── Hero ──
    st.markdown("""<div class="land-hero">
        <div class="logo-shimmer">TRAXR</div>
        <div class="land-tagline">TRACK · PROVE · ADVANCE</div>
        <p class="land-sub">TRAXR shows students exactly what skills they are missing for a specific internship
        and gives them a proof-first roadmap to close the gap.</p>
    </div>""", unsafe_allow_html=True)

    # CTA buttons
    _, c1, c2, _ = st.columns([1.5, 1, 1, 1.5])
    with c1:
        st.markdown('<div class="cta-primary">', unsafe_allow_html=True)
        if st.button("🚀 Start Analysis", key="cta_start", use_container_width=True):
            st.session_state.view = "dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cta-ghost">', unsafe_allow_html=True)
        if st.button("▶ Try Demo", key="cta_demo", use_container_width=True):
            st.session_state.view = "dashboard"
            st.session_state.using_demo = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Preview Card ──
    _, prev_col, _ = st.columns([1, 2, 1])
    with prev_col:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        pc1, pc2 = st.columns([1, 1.2])
        with pc1:
            score_circle(demodata.DEMO_SCORES["overall_score"])
        with pc2:
            st.plotly_chart(build_radar(demodata.DEMO_SCORES["dimensions"], h=280), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Features ──
    st.markdown('<div class="land-section" style="animation-delay:.3s"><h2>Why TRAXR?</h2><p class="sec-sub">Not another ATS checker. A skill credibility system built for students.</p></div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    feats = [
        ("🎯","Role-Specific Analysis","Extracts the DNA of each job description — required skills, preferred skills, tools, and behavioral signals."),
        ("📊","360° Readiness Score","Six transparent dimensions of job readiness. Every score is explainable — no black boxes."),
        ("🧩","Proof-First System","Distinguishes between skills you claim and skills you can prove. SkillPrint lets you demonstrate live."),
    ]
    for i, (icon, title, desc) in enumerate(feats):
        with [f1,f2,f3][i]:
            st.markdown(f'<div class="feat-card" style="animation-delay:{.3+i*.15}s"><span class="fi">{icon}</span><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown('<div class="land-section" style="animation-delay:.5s"><h2>How It Works</h2><p class="sec-sub">From resume to roadmap in four intelligent steps.</p></div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    steps = [
        ("1","Upload & Paste","Upload your resume PDF and paste the job description."),
        ("2","Role DNA","AI extracts what the role truly requires — beyond keywords."),
        ("3","360° Score","Deterministic scoring across 6 dimensions with gap classification."),
        ("4","Roadmap","Personalized 30-day plan with free resources and proof milestones."),
    ]
    for i, (num, title, desc) in enumerate(steps):
        with [s1,s2,s3,s4][i]:
            st.markdown(f'<div class="step-card" style="animation-delay:{.5+i*.12}s"><div class="step-num">{num}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── 6 Dimensions ──
    st.markdown('<div class="land-section" style="animation-delay:.7s"><h2>6 Readiness Dimensions</h2><p class="sec-sub">A transparent model — every score is earned, every gap explained.</p></div>', unsafe_allow_html=True)
    dims_info = [("💻","Technical Match"),("📋","Evidence Quality"),("💬","Communication"),("🎯","Role Alignment"),("📈","Momentum"),("🎤","Interview Ready")]
    d_cols = st.columns(6, gap="small")
    for i,(icon,name) in enumerate(dims_info):
        with d_cols[i]:
            st.markdown(f'<div class="dim-card-sm" style="animation-delay:{.7+i*.1}s"><div class="di">{icon}</div><h4>{name}</h4></div>', unsafe_allow_html=True)

    st.markdown("""<br><br><div style="text-align:center;color:var(--t3);font-size:.82rem;padding:2rem 0">
        TRAXR v1.0 · Built for DevYatra 2026</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════
def render_dashboard():
    # ── Header ──
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    name = ""
    if st.session_state.analysis_complete and st.session_state.using_demo:
        name = ", Priya"

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(f"""<div class="dash-hdr"><div class="dash-hdr-left">
            <span class="logo-sm">TRAXR</span>
            <span class="dash-greeting">{greet}{name}</span>
        </div></div>""", unsafe_allow_html=True)
    with hdr_r:
        if st.button("← Back", key="back_landing"):
            st.session_state.view = "landing"
            st.rerun()

    # ── Auto-run demo if flagged ──
    if st.session_state.using_demo and not st.session_state.analysis_complete:
        _run_analysis(demodata.DEMO_RESUME_TEXT, demodata.DEMO_JD_TEXT, True)

    # ── Input Section ──
    if not st.session_state.analysis_complete:
        _render_inputs()
    else:
        with st.expander("📝 Change inputs", expanded=False):
            _render_inputs()

    # ── Results ──
    if st.session_state.analysis_complete and st.session_state.scores:
        _render_results()

    # Footer
    st.markdown("""<div style="text-align:center;color:var(--t3);font-size:.8rem;padding:2rem 0">
        TRAXR v1.0 · DevYatra 2026 · Track. Prove. Advance.</div>""", unsafe_allow_html=True)


def _render_inputs():
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="d-card"><div class="d-card-title">📄 Resume</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed", key="pdf_up")
        resume_paste = st.text_area("Or paste resume text", height=100, placeholder="Paste resume text…", label_visibility="collapsed", key="res_paste")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="d-card"><div class="d-card-title">📋 Job Description</div>', unsafe_allow_html=True)
        jd_input = st.text_area("Paste JD", height=150, placeholder="Paste the full job description…", label_visibility="collapsed", key="jd_in")
        if jd_input and len(jd_input) < 100:
            st.warning(f"JD seems short ({len(jd_input)} chars).")
        st.markdown('</div>', unsafe_allow_html=True)

    _, btn_c, _ = st.columns([2, 1.5, 2])
    with btn_c:
        st.markdown('<div class="cta-primary">', unsafe_allow_html=True)
        if st.button("🚀 Analyze My Readiness", use_container_width=True, key="analyze_btn"):
            raw = ""
            using_demo = False
            if uploaded:
                raw = extract_pdf_text(uploaded)
            if not raw and resume_paste:
                raw = resume_paste.strip()
            if not raw:
                raw = demodata.DEMO_RESUME_TEXT; using_demo = True
            jd = jd_input.strip() if jd_input else ""
            if not jd:
                jd = demodata.DEMO_JD_TEXT; using_demo = True
            _run_analysis(raw, jd, using_demo)
        st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.markdown("""<div style="text-align:center;padding:3rem 0;color:var(--t3)">
            <div style="font-size:3rem;margin-bottom:.5rem">📊</div>
            <p style="font-size:1rem;font-weight:500">Upload your resume and paste a job description to begin</p>
            <p style="font-size:.85rem">Or click Analyze to try with demo data</p></div>""", unsafe_allow_html=True)


def _run_analysis(raw, jd, using_demo):
    new_hash = _h(raw + jd)
    if new_hash == st.session_state.input_hash and st.session_state.analysis_complete:
        return
    with st.spinner("🔍 Extracting Role DNA…"):
        role_dna = extract_role_dna(jd)
    with st.spinner("📊 Analyzing profile…"):
        signals = extract_profile_signals(raw)
        scores = compute_360_score(raw, role_dna, signals)
    with st.spinner("🗺️ Generating roadmap…"):
        gaps = scores["ranked_gaps"]
        roadmap = generate_roadmap(gaps, role_dna.get("role_context", ""))
    challenge = None
    top_hard = next((g for g in gaps if g["type"] == "hard"), None)
    if top_hard:
        with st.spinner("🧩 Creating SkillPrint…"):
            challenge = generate_skillprint_challenge(top_hard["skill"])
    st.session_state.update({
        "raw_text": raw, "jd_text": jd, "input_hash": new_hash,
        "using_demo": using_demo, "role_dna": role_dna,
        "profile_signals": signals.to_dict() if hasattr(signals, 'to_dict') else signals,
        "scores": scores, "gaps": gaps, "roadmap": roadmap,
        "skillprint_challenge": challenge, "skillprint_code": "",
        "skillprint_result": None, "analysis_complete": True,
    })
    st.rerun()


def _render_results():
    scores = st.session_state.scores
    dims = scores["dimensions"]
    expl = scores["dimension_explanations"]
    gaps = st.session_state.gaps
    roadmap = st.session_state.roadmap

    if st.session_state.using_demo:
        st.info("🔔 **Demo mode** — showing sample analysis. Upload your own resume & JD for personalized results.")

    # ── Score Hero Row ──
    sc_col, radar_col = st.columns([1, 1.4], gap="large")
    with sc_col:
        st.markdown('<div class="d-card" style="animation-delay:.1s">', unsafe_allow_html=True)
        score_circle(scores["overall_score"])
        st.markdown(f'<div style="text-align:center;margin-top:.6rem;color:var(--t3);font-size:.82rem">Confidence: <b style="color:var(--t2)">{scores.get("confidence","—").title()}</b></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="d-card" style="animation-delay:.2s"><div class="d-card-title">Dimension Scores</div>', unsafe_allow_html=True)
        for i, (d, s) in enumerate(dims.items()):
            pbar(d, s, delay=.3 + i * .1)
        st.markdown('</div>', unsafe_allow_html=True)

    with radar_col:
        st.markdown('<div class="d-card" style="animation-delay:.15s"><div class="d-card-title">360° Readiness Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(build_radar(dims), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    t_ov, t_dna, t_bd, t_gap, t_sp, t_road = st.tabs(
        ["Overview", "Role DNA", "Score Breakdown", "Top Gaps", "SkillPrint", "Roadmap"])

    with t_ov:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Score", f"{scores['overall_score']}/100")
        m2.metric("Matched", len(st.session_state.profile_signals.get("claimed_skills",[])))
        m3.metric("Gaps", len(gaps))
        m4.metric("Confidence", scores.get("confidence","—").title())
        mc, gc = st.columns(2, gap="medium")
        matched_html = "".join(f'<span class="chip chip-matched">{s.title()}</span>' for s in st.session_state.profile_signals.get("supported_skills",[])[:12])
        gap_html = "".join(f'<span class="chip chip-{g["type"]}">{g["skill"].title()}</span>' for g in gaps[:8])
        with mc:
            st.markdown(f'<div class="d-card"><div class="d-card-title">✅ Matched Skills</div>{matched_html or "<span style=color:var(--t3)>None detected</span>"}</div>', unsafe_allow_html=True)
        with gc:
            st.markdown(f'<div class="d-card"><div class="d-card-title">⚠️ Skill Gaps</div>{gap_html or "<span style=color:var(--ok)>No gaps found!</span>"}</div>', unsafe_allow_html=True)

    with t_dna:
        dna = st.session_state.role_dna or {}
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(f'<div class="d-card"><div class="d-card-title">🔴 Required Skills</div><p style="color:var(--t2)">{", ".join(s.title() for s in dna.get("required_skills",[])) or "—"}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="d-card"><div class="d-card-title">🟡 Preferred Skills</div><p style="color:var(--t2)">{", ".join(s.title() for s in dna.get("preferred_skills",[])) or "—"}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="d-card"><div class="d-card-title">🛠️ Tools</div><p style="color:var(--t2)">{", ".join(s.title() for s in dna.get("tools",[])) or "—"}</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="d-card"><div class="d-card-title">🎯 Role Context</div><p style="color:var(--t2)">{dna.get("role_context","—")}</p></div>', unsafe_allow_html=True)
            acts = "".join(f"<li>{a}</li>" for a in dna.get("day_to_day_activities",[]))
            st.markdown(f'<div class="d-card"><div class="d-card-title">💼 Day-to-Day</div><ul style="color:var(--t2);padding-left:1.2rem;margin:0">{acts or "<li>—</li>"}</ul></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="d-card"><div class="d-card-title">🧠 Behavioral</div><p style="color:var(--t2)">{", ".join(dna.get("behavioral_requirements",[])) or "—"}</p></div>', unsafe_allow_html=True)

    with t_bd:
        for d_name in dims:
            e = expl.get(d_name, {})
            emoji = "🟢" if e.get("score",0) >= 75 else "🟡" if e.get("score",0) >= 50 else "🔴"
            with st.expander(f"{emoji} {d_name} — {e.get('score',0)}/100", expanded=False):
                st.markdown(f"**Why this score?** {e.get('why','—')}")
                ev = e.get("evidence_found",[])
                if ev: st.markdown("**Evidence:** " + " · ".join(str(x) for x in ev[:6]))
                mi = e.get("missing",[])
                if mi: st.markdown("**Missing:** " + " · ".join(str(x) for x in mi[:5]))
                st.markdown(f"**Action:** {e.get('action','—')}")

    with t_gap:
        if not gaps:
            st.success("🎉 No significant gaps — you're a strong fit!")
        for g in gaps:
            se = {"high":"🔴","medium":"🟡","low":"🟢"}.get(g["severity"],"⚪")
            tl = {"hard":"Hard Gap","soft":"Soft Gap","context":"Context Gap"}.get(g["type"],"Gap")
            with st.expander(f'{se} {g["skill"].title()} — {tl} ({g["severity"]})', expanded=(g["severity"]=="high")):
                st.markdown(f'**Category:** {g.get("category","—")}')
                st.markdown(f'**Why it matters:** {g["explanation"]}')
                st.markdown(f'**Action:** {g["action"]}')

    with t_sp:
        ch = st.session_state.skillprint_challenge
        if not ch:
            st.info("No SkillPrint challenge — all critical skills are matched!")
        else:
            st.markdown(f'<div class="d-card"><div class="d-card-title">🧩 {ch["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'**Skill:** `{ch["skill"]}` · **Time:** {ch.get("time_estimate","5 min")}')
            st.markdown(ch.get("description",""))
            for r in ch.get("rubric",[]): st.markdown(f"- {r}")
            st.markdown('</div>', unsafe_allow_html=True)
            code = st.text_area("✍️ Your solution", value=st.session_state.skillprint_code or ch.get("starter_code",""), height=220, key="sp_code")
            st.session_state.skillprint_code = code
            if st.button("🚀 Submit", key="sp_sub"):
                with st.spinner("Evaluating…"):
                    result = evaluate_skillprint(code, ch)
                    st.session_state.skillprint_result = result
                    if result.get("passed") and st.session_state.scores:
                        boost = ch.get("max_score_boost", 8)
                        st.session_state.scores["dimensions"]["Technical Skill Match"] = min(100, st.session_state.scores["dimensions"]["Technical Skill Match"] + boost)
                        d = st.session_state.scores["dimensions"]
                        st.session_state.scores["overall_score"] = round(sum(d.values()) / len(d))
                        st.rerun()
            result = st.session_state.skillprint_result
            if result:
                rc = _sc_hex(result.get("score",0))
                st.markdown(f'<div class="d-card"><div class="d-card-title" style="color:{rc}">Score: {result.get("score",0)}/100 {"✅ Passed!" if result.get("passed") else "❌ Not yet"}</div>', unsafe_allow_html=True)
                st.markdown(f'**Feedback:** {result.get("feedback","—")}')
                for imp in result.get("improvements",[]): st.markdown(f"- {imp}")
                st.markdown('</div>', unsafe_allow_html=True)

    with t_road:
        if not roadmap:
            st.info("Roadmap will appear after analysis.")
        else:
            for week in roadmap:
                st.markdown(f"### 📅 Week {week.get('week','?')} — {week.get('title','')}")
                for db in week.get("days",[]):
                    tasks_html = "".join(f"<li>{t}</li>" for t in db.get("tasks",[]))
                    st.markdown(f'<div class="road-block"><h4>Day {db.get("day","")} — {db.get("goal","")}</h4><ul>{tasks_html}</ul></div>', unsafe_allow_html=True)
                    for r in db.get("resources",[]):
                        if isinstance(r,dict): st.markdown(f'📎 [{r.get("name","")}]({r.get("url","")})')
                        else: st.markdown(f"📎 {r}")
                    if db.get("proof"): st.caption(f"✅ Proof: {db['proof']}")


# ═══════════════════════════════════════════════════════════
# PAGE ROUTER
# ═══════════════════════════════════════════════════════════
if st.session_state.view == "landing":
    render_landing()
else:
    render_dashboard()