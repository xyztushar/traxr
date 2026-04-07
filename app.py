"""
TRAXR — Track. Prove. Advance.
Streamlit app: Landing → Dashboard with Gemini-powered 360° readiness analysis.
"""
from __future__ import annotations
import hashlib
import os
import streamlit as st
import plotly.graph_objects as go
import demodata
from models import AnalysisResult, RoleDNA, ScoreDimension, GapItem, SkillPrint, RoadmapWeek
from nlppipeline import (
    extract_pdf_text_basic, is_text_low_quality,
    render_pdf_pages_for_ocr, extract_text_with_gemini_from_pages,
    normalize_resume_text,
    extract_skills, extract_profile_signals,
)
from scorer import compute_360_score, classify_gaps
from llmclient import (
    build_role_dna, build_score_breakdown, build_gaps,
    build_skillprint, build_roadmap, evaluate_skillprint_submission, is_llm_available,
)

st.set_page_config(page_title="TRAXR – Track. Prove. Advance.", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*{font-family:'Inter',system-ui,sans-serif}

:root{
  --bg:#060912;--bg2:#0c1120;--surface:#111827;--card:#151e32;--card-h:#1a2540;
  --bdr:rgba(148,163,184,.07);--bdr-h:rgba(148,163,184,.14);--bdr-acc:rgba(99,102,241,.2);
  --acc:#6366f1;--acc2:#818cf8;--acc3:#a5b4fc;--glow:rgba(99,102,241,.08);
  --t1:#f1f5f9;--t2:#94a3b8;--t3:#64748b;--t4:#475569;
  --ok:#34d399;--warn:#fbbf24;--err:#f87171;
  --r:14px;--r-sm:8px;--r-lg:20px;
  --s1:.25rem;--s2:.5rem;--s3:.75rem;--s4:1rem;--s5:1.5rem;--s6:2rem;--s7:2.5rem;--s8:3rem;
}

.stApp{background:var(--bg)!important;color:var(--t1)}
.stApp::before{content:'';position:fixed;inset:0;
  background:
    radial-gradient(ellipse 900px 600px at 10% 20%,rgba(99,102,241,.05),transparent),
    radial-gradient(ellipse 700px 500px at 80% 70%,rgba(79,70,229,.04),transparent),
    radial-gradient(ellipse 500px 400px at 50% 100%,rgba(99,102,241,.03),transparent);
  pointer-events:none;z-index:0}
.block-container{max-width:1140px!important;padding-top:1rem!important;padding-bottom:0!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{display:none}
hr{border-color:var(--bdr)!important}

/* Typography */
.t-brand{font-size:4rem;font-weight:900;letter-spacing:-2.5px;line-height:1;
  background:linear-gradient(135deg,#6366f1 0%,#a78bfa 25%,#e0e7ff 50%,#a78bfa 75%,#6366f1 100%);
  background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shimmer 5s linear infinite;display:inline-block}
.t-brand-sm{font-size:1.4rem;letter-spacing:-1px;font-weight:800;
  background:linear-gradient(135deg,#6366f1,#a78bfa,#6366f1);background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 5s linear infinite}
.t-body{font-size:.92rem;color:var(--t2);line-height:1.6}
.t-caption{font-size:.78rem;color:var(--t3);line-height:1.5}

/* Animations */
@keyframes shimmer{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes fillBar{from{width:0}}
@keyframes typeReveal{from{max-width:0;opacity:0}to{max-width:600px;opacity:1}}

/* Cards */
.card{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);padding:var(--s5);
  animation:fadeUp .45s ease both}
.card--interactive{transition:border-color .2s,background .2s}
.card--interactive:hover{border-color:var(--bdr-acc);box-shadow:0 0 0 1px var(--bdr-acc)}
.card-title{font-size:.82rem;font-weight:700;color:var(--t3);text-transform:uppercase;
  letter-spacing:.8px;margin-bottom:var(--s3);display:flex;align-items:center;gap:.4rem}

/* Preview (Landing) */
.preview-frame{background:var(--surface);border:1px solid var(--bdr-acc);border-radius:var(--r-lg);
  overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.4),0 0 0 1px rgba(99,102,241,.08);
  animation:fadeUp .6s ease .3s both}
.preview-bar{background:var(--bg2);padding:.45rem .8rem;display:flex;align-items:center;gap:.4rem;
  border-bottom:1px solid var(--bdr)}
.preview-dot{width:7px;height:7px;border-radius:50%;background:var(--t4)}
.preview-dot:nth-child(1){background:#f87171}.preview-dot:nth-child(2){background:#fbbf24}
.preview-dot:nth-child(3){background:#34d399}
.preview-body{padding:var(--s5)}
.preview-score{display:flex;align-items:center;gap:var(--s5);margin-bottom:var(--s4)}
.preview-ring{width:72px;height:72px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.preview-ring-inner{width:58px;height:58px;border-radius:50%;background:var(--surface);
  display:flex;align-items:center;justify-content:center;font-size:1.3rem;font-weight:900}
.preview-dims{flex:1}
.preview-dim{display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem}
.preview-dim-label{font-size:.65rem;color:var(--t3);width:80px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.preview-dim-track{flex:1;height:4px;background:rgba(30,42,69,.8);border-radius:3px;overflow:hidden}
.preview-dim-fill{height:100%;border-radius:3px}
.preview-chip{font-size:.55rem;padding:2px 7px;border-radius:10px;font-weight:600;color:#fff}

/* Landing */
.land-wrap{animation:fadeIn .6s ease}
.land-hero-left{display:flex;flex-direction:column;justify-content:center;min-height:420px;padding:var(--s6) 0}
.land-tagline{font-size:.82rem;font-weight:700;color:var(--acc2);text-transform:uppercase;
  letter-spacing:2.5px;margin-bottom:var(--s3)}
.land-headline{font-size:1.75rem;font-weight:800;color:var(--t1);line-height:1.3;
  letter-spacing:-.5px;margin-bottom:var(--s4)}
.land-desc{font-size:.95rem;color:var(--t2);line-height:1.65;margin-bottom:var(--s6);max-width:440px}
.land-value{display:flex;gap:var(--s6);margin-top:var(--s5);flex-wrap:wrap}
.land-value-item{display:flex;align-items:center;gap:.4rem;font-size:.78rem;color:var(--t3);font-weight:500}
.land-footer{text-align:center;color:var(--t4);font-size:.75rem;padding:var(--s6) 0 var(--s4);
  border-top:1px solid var(--bdr);margin-top:var(--s7)}
[data-testid="stHorizontalBlock"]{align-items:stretch}

/* ── Feature Showcase ── */
.showcase{padding:4.5rem 0 1rem;animation:fadeUp .5s ease .2s both}
.showcase-eyebrow{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:3px;
  color:var(--acc2);margin-bottom:var(--s3);text-align:center}
.showcase-headline{font-size:1.6rem;font-weight:800;letter-spacing:-.5px;line-height:1.3;
  text-align:center;margin-bottom:var(--s2);
  background:linear-gradient(135deg,#e0e7ff 0%,#a78bfa 50%,#818cf8 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;display:block}
.showcase-sub{font-size:.88rem;color:var(--t3);text-align:center;margin-bottom:2.5rem;
  max-width:500px;margin-left:auto;margin-right:auto;line-height:1.55}

/* Feature Card — premium structure */
.feature-card{background:#111827;border:1px solid #1e2d3d;border-radius:16px;
  padding:2.5rem 2rem;height:100%;display:flex;flex-direction:column;
  justify-content:space-between;transition:all .3s ease}
.feature-card:hover{transform:translateY(-4px);border-color:#3b82f6;
  box-shadow:0 12px 32px -12px rgba(59,130,246,.25)}

.fc-content-top{display:flex;flex-direction:column;gap:1rem;width:100%}
.fc-content-bottom{margin-top:auto;padding-top:2rem;
  width:100%!important;display:flex;flex-direction:column;
  justify-content:flex-end;align-items:stretch!important}

.fc-visual{display:flex;align-items:center;gap:8px}
.fc-icon{width:42px;height:42px;border-radius:10px;display:flex;align-items:center;
  justify-content:center;font-size:1.2rem;flex-shrink:0;
  background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(124,58,237,.08));
  border:1px solid rgba(99,102,241,.12)}
.fc-badge{font-size:.58rem;font-weight:700;padding:2px 8px;border-radius:6px;
  text-transform:uppercase;letter-spacing:.5px}
.fc-badge--req{color:#818cf8;background:rgba(129,140,248,.1);border:1px solid rgba(129,140,248,.12)}
.fc-badge--pref{color:#a78bfa;background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.1)}
.fc-badge--ctx{color:#64748b;background:rgba(100,116,139,.1);border:1px solid rgba(100,116,139,.12)}

.fc-title{font-size:.95rem;font-weight:700;color:var(--t1);line-height:1.3;transition:color .2s}
.feature-card:hover .fc-title{color:var(--acc3)}
.fc-desc{font-size:.8rem;color:var(--t3);line-height:1.6;margin:0}

.fc-mini-bars,.mini-bar-wrapper{display:flex;flex-direction:column;gap:4px;
  width:100%!important;max-width:100%!important;display:block!important;
  box-sizing:border-box!important;margin-bottom:0!important}
.fc-mini-bars{display:flex!important;flex-direction:column;gap:4px}
.fc-mini-bar{display:flex;align-items:center;gap:6px}
.fc-mini-label{font-size:.58rem;color:var(--t4);width:58px;text-overflow:ellipsis;
  overflow:hidden;white-space:nowrap;font-weight:600}
.fc-mini-track{flex:1;height:3px;background:rgba(30,42,69,.8);border-radius:2px;overflow:hidden}
.fc-mini-fill{height:100%;border-radius:2px}

.fc-terminal{background:#0a0f1a;border:1px solid rgba(148,163,184,.06);border-radius:8px;
  padding:12px 14px;font-family:'Courier New',monospace;font-size:.68rem;
  color:var(--t3);line-height:1.7;overflow:hidden;
  width:100%!important;max-width:100%!important;display:block!important;
  box-sizing:border-box!important;margin-bottom:0!important}
.fc-terminal .t-kw{color:#818cf8}
.fc-terminal .t-fn{color:#34d399}
.fc-terminal .t-str{color:#fbbf24}
.fc-terminal .t-cmt{color:#475569;font-style:italic}

.fc-timeline{display:flex;gap:4px;align-items:flex-end;
  width:100%!important;max-width:100%!important;
  box-sizing:border-box!important;margin-bottom:0!important}
.fc-tl-bar{flex:1;border-radius:3px 3px 0 0;min-width:0;
  background:linear-gradient(to top,rgba(99,102,241,.15),rgba(99,102,241,.3));
  border:1px solid rgba(99,102,241,.1);transition:background .3s}
.feature-card:hover .fc-tl-bar{background:linear-gradient(to top,rgba(99,102,241,.2),rgba(99,102,241,.45))}
.fc-tl-labels{display:flex;justify-content:space-between;margin-top:4px}
.fc-tl-label{font-size:.54rem;color:var(--t4);font-weight:600}

/* Dashboard header */
.dash-bar{display:flex;align-items:center;justify-content:space-between;
  padding:var(--s2) 0 var(--s4);border-bottom:1px solid var(--bdr);margin-bottom:var(--s5)}
.dash-bar-left{display:flex;align-items:center;gap:var(--s3)}
.dash-bar-tag{font-size:.75rem;font-weight:600;color:var(--acc2);background:var(--glow);
  padding:2px 10px;border-radius:20px;border:1px solid var(--bdr-acc)}

/* Greeting */
.greeting{font-size:1.05rem;font-weight:600;color:var(--acc3);margin-bottom:var(--s5);
  overflow:hidden;white-space:nowrap;display:inline-block}
.greeting--anim{animation:typeReveal .8s ease forwards}
.greeting--static{max-width:600px;opacity:1}

/* Gemini indicator */
.engine-badge{display:inline-flex;align-items:center;gap:.3rem;font-size:.72rem;font-weight:600;
  padding:3px 10px;border-radius:12px;margin-left:var(--s3)}
.engine-badge--gemini{color:#34d399;background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.15)}
.engine-badge--local{color:var(--t3);background:rgba(100,116,139,.08);border:1px solid rgba(100,116,139,.12)}

/* Score ring */
.score-ring{width:150px;height:150px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;margin:0 auto var(--s3)}
.score-ring-inner{width:122px;height:122px;border-radius:50%;background:var(--card);
  display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:1}
.score-ring-val{font-size:2.6rem;font-weight:900;line-height:1}
.score-ring-sub{font-size:.68rem;color:var(--t3);margin-top:2px;font-weight:500}

/* Progress bars */
.pbar{margin:var(--s2) 0}
.pbar-row{display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:3px}
.pbar-name{color:var(--t2);font-weight:500}
.pbar-val{font-weight:700}
.pbar-track{background:rgba(30,42,69,.8);border-radius:4px;height:6px;overflow:hidden}
.pbar-fill{height:100%;border-radius:4px;animation:fillBar 1s ease both}

/* Chips */
.chip{display:inline-block;padding:3px 10px;margin:2px;border-radius:20px;font-size:.74rem;
  font-weight:600;color:#fff;transition:transform .15s,filter .15s}
.chip:hover{transform:translateY(-1px);filter:brightness(1.1)}
.chip-hard{background:var(--err)}.chip-soft{background:var(--warn);color:#1a1a1a}
.chip-context{background:var(--acc)}.chip-matched{background:var(--ok);color:#1a1a1a}

/* Roadmap */
.road-item{background:var(--card);border-left:3px solid var(--acc);border-radius:0 var(--r-sm) var(--r-sm) 0;
  padding:var(--s4) var(--s5);margin-bottom:var(--s3);transition:border-color .2s,background .2s}
.road-item:hover{border-left-color:var(--acc2);background:var(--card-h)}
.road-item h4{color:var(--acc2);margin:0 0 .3rem;font-size:.88rem;font-weight:700}
.road-item ul{margin:0;padding-left:1.1rem;color:var(--t2);font-size:.82rem;line-height:1.6}

/* Overrides */
.stTextArea textarea{background:var(--bg2)!important;color:var(--t1)!important;
  border:1px solid var(--bdr)!important;border-radius:var(--r-sm)!important;font-size:.88rem!important}
.stTextArea textarea:focus{border-color:var(--acc)!important;box-shadow:none!important}
.stTextArea [data-baseweb="textarea"]{border-color:var(--bdr)!important}
.stTextInput input{background:var(--bg2)!important;color:var(--t1)!important;
  border:1px solid var(--bdr)!important;border-radius:var(--r-sm)!important}
.stTextInput input:focus{border-color:var(--acc)!important;box-shadow:none!important}
div[data-testid="stFileUploader"]{background:var(--bg2);border:1px dashed var(--bdr-acc);
  border-radius:var(--r-sm);padding:.7rem;transition:border-color .2s}
div[data-testid="stFileUploader"]:hover{border-color:var(--acc)}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#6366f1,#7c3aed)!important;color:#fff!important;
  border:none!important;border-radius:var(--r-sm)!important;padding:.55rem 1.6rem!important;
  font-weight:700!important;font-size:.85rem!important;transition:all .2s ease!important;white-space:nowrap!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(99,102,241,.35)!important}
.stButton>button:active{transform:translateY(0)!important}
.cta-primary .stButton>button{font-size:.95rem!important;padding:.7rem 2rem!important;border-radius:10px!important;
  white-space:nowrap!important}
.cta-ghost .stButton>button{background:transparent!important;border:1.5px solid var(--bdr-h)!important;
  color:var(--t2)!important;box-shadow:none!important;white-space:nowrap!important}
.cta-ghost .stButton>button:hover{border-color:var(--acc)!important;color:var(--t1)!important;
  background:var(--glow)!important;transform:translateY(-2px)!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--bdr);padding:0}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--t4);border:none;
  border-bottom:2px solid transparent;border-radius:0;padding:.55rem 1rem;
  font-weight:600;font-size:.78rem;transition:color .2s,background .15s}
.stTabs [data-baseweb="tab"]:hover{color:var(--t1);background:var(--glow)}
.stTabs [aria-selected="true"]{color:var(--acc2)!important;border-bottom-color:var(--acc)!important;
  background:transparent!important}

/* Expander */
.streamlit-expanderHeader{background:var(--card)!important;border-radius:var(--r-sm)!important;
  font-weight:600!important;border:1px solid var(--bdr)!important;font-size:.88rem!important}
.streamlit-expanderHeader:hover{border-color:var(--bdr-h)!important;background:var(--card-h)!important}

/* Metrics */
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r-sm);
  padding:.7rem .8rem;text-align:center}
[data-testid="stMetricLabel"]{color:var(--t4)!important;font-weight:600!important;font-size:.72rem!important}
[data-testid="stMetricValue"]{color:var(--t1)!important;font-weight:800!important;font-size:1.3rem!important}

/* Input Command Center */
.input-card-wrap{margin-bottom:1.5rem}
.input-card-wrap [data-testid="stVerticalBlockBorderWrapper"]{
  background:rgba(17,24,39,.4)!important;border:1px solid #1e2d3d!important;
  border-radius:16px!important;padding:1.5rem!important;transition:all .3s ease}
.input-card-wrap [data-testid="stVerticalBlockBorderWrapper"]:hover{
  border-color:var(--bdr-h)!important}
.input-card-wrap [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"]{
  background:transparent!important;border:none!important;box-shadow:none!important;
  padding:0!important;border-radius:0!important;height:auto!important}
.input-card-ready [data-testid="stVerticalBlockBorderWrapper"]{
  border-color:#3b82f6!important;background:rgba(17,24,39,.6)!important;
  box-shadow:0 0 20px rgba(59,130,246,.1)}
.input-card-label{font-family:'Inter',system-ui,sans-serif;color:#94a3b8;font-size:.75rem;
  letter-spacing:.15rem;text-transform:uppercase;margin-bottom:1.25rem;
  display:flex;align-items:center;gap:.6rem;font-weight:700}
.or-divider{display:flex;align-items:center;text-align:center;color:#4b5563;
  font-size:.65rem;font-weight:600;margin:1.25rem 0;letter-spacing:.1rem}
.or-divider::before,.or-divider::after{content:'';flex:1;border-bottom:1px solid #1e2d3d}
.or-divider:not(:empty)::before{margin-right:1rem}
.or-divider:not(:empty)::after{margin-left:1rem}
.helper-text{color:#64748b;font-size:.7rem;line-height:1.4;margin-top:.5rem}
.analyze-cta-row{display:flex;justify-content:center;margin-top:2rem;width:100%}
/* Dark-style file uploader */
[data-testid="stFileUploader"]{background:transparent}
[data-testid="stFileUploaderDropzone"]{background:rgba(30,45,61,.45)!important;
  border:1px dashed rgba(148,163,184,.18)!important;border-radius:.75rem!important}
[data-testid="stFileUploaderDropzone"]:hover{border-color:rgba(99,102,241,.4)!important;
  background:rgba(30,45,61,.65)!important}
[data-testid="stFileUploaderDropzone"] small{color:#64748b!important}
[data-testid="stFileUploaderDropzone"] button{background:#1e293b!important;
  color:#94a3b8!important;border:1px solid rgba(148,163,184,.15)!important}

/* Demo Lab — Compact premium banner */
@keyframes demoBorderGlow{
  0%,100%{border-color:rgba(251,191,36,.15);box-shadow:0 0 20px rgba(251,191,36,.03)}
  50%{border-color:rgba(167,139,250,.25);box-shadow:0 0 28px rgba(99,102,241,.06)}
}
@keyframes demoOrbFloat{
  0%,100%{transform:translateY(0)}
  50%{transform:translateY(-4px)}
}
.demo-banner{position:relative;overflow:hidden;border-radius:16px;padding:1.2rem 1.5rem 1rem;
  text-align:center;margin-bottom:1rem;
  background:linear-gradient(145deg,rgba(17,24,39,.85),rgba(15,23,42,.95));
  border:1px solid rgba(251,191,36,.12);
  animation:fadeUp .4s ease both, demoBorderGlow 4s ease-in-out infinite}
.demo-banner::before{content:'';position:absolute;top:-60%;left:-30%;width:160%;height:160%;
  background:radial-gradient(ellipse at 30% 20%,rgba(251,191,36,.05),transparent 50%),
    radial-gradient(ellipse at 70% 80%,rgba(99,102,241,.04),transparent 50%);
  pointer-events:none}
.demo-banner-icon{font-size:1.3rem;margin-right:.5rem;vertical-align:middle;
  animation:demoOrbFloat 3s ease-in-out infinite;display:inline-block}
.demo-banner-title{font-size:1.05rem;font-weight:900;letter-spacing:-.3px;display:inline;
  background:linear-gradient(135deg,#fbbf24 0%,#f59e0b 30%,#e0e7ff 60%,#a78bfa 100%);
  background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shimmer 4s linear infinite}
.demo-banner-row{display:flex;align-items:center;justify-content:center;gap:.1rem;
  margin-bottom:.4rem}
.demo-banner-sub{font-size:.78rem;color:var(--t3);max-width:420px;margin:0 auto .6rem;line-height:1.5}
.demo-flow-steps{display:flex;justify-content:center;gap:1rem}
.demo-flow-step{display:flex;align-items:center;gap:.3rem;font-size:.62rem;color:var(--t4);
  font-weight:600;letter-spacing:.3px}
.demo-flow-step span{width:18px;height:18px;border-radius:50%;display:inline-flex;
  align-items:center;justify-content:center;font-size:.55rem;font-weight:800;
  background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.2)}
.demo-flow-arrow{color:var(--t4);font-size:.55rem;opacity:.4}

/* Demo generate CTA — main (gold gradient border) */
.demo-gen-cta{position:relative;border-radius:12px;padding:2px;
  background:linear-gradient(135deg,rgba(251,191,36,.4),rgba(167,139,250,.4),rgba(251,191,36,.4));
  background-size:200% auto;animation:shimmer 3s linear infinite;
  margin:0 auto;max-width:340px;transition:all .3s ease}
.demo-gen-cta:hover{transform:translateY(-2px);
  box-shadow:0 6px 24px rgba(251,191,36,.15),0 3px 12px rgba(99,102,241,.1)}

/* Demo generate — individual column buttons (indigo gradient border) */
.demo-gen-sm{position:relative;border-radius:10px;padding:1.5px;
  background:linear-gradient(135deg,rgba(99,102,241,.35),rgba(167,139,250,.3),rgba(99,102,241,.35));
  background-size:200% auto;animation:shimmer 3.5s linear infinite;
  transition:all .3s ease}
.demo-gen-sm:hover{transform:translateY(-2px);
  box-shadow:0 5px 20px rgba(99,102,241,.15),0 2px 10px rgba(167,139,250,.1)}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
_DEFAULTS = {
    "view": "landing",
    "user_name": "",
    "result": None,           # AnalysisResult | None
    "skillprint_code": "",
    "skillprint_eval": None,
    "greeting_shown": False,  # Prevents greeting animation replay
    "demo_generated_resume": "",
    "demo_generated_jd": "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def _h(t): return hashlib.md5(t.encode()).hexdigest()

def _get_api_key() -> str | None:
    """Resolve the next available Gemini API key from the rotation pool."""
    from key_pool import get_next_api_key
    return get_next_api_key()

def _color(s):
    if s >= 75: return "#34d399"
    if s >= 50: return "#fbbf24"
    return "#f87171"

def build_radar(dims, h=360):
    cats = list(dims.keys()); vals = list(dims.values())
    cats += [cats[0]]; vals += [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
        fillcolor="rgba(99,102,241,0.1)", line=dict(color="#818cf8", width=2),
        marker=dict(size=4, color="#818cf8")))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(color="#475569",size=8),
                gridcolor="rgba(148,163,184,.06)", linecolor="rgba(148,163,184,.04)"),
            angularaxis=dict(tickfont=dict(color="#94a3b8",size=9),
                gridcolor="rgba(148,163,184,.06)", linecolor="rgba(148,163,184,.04)")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(l=70,r=70,t=25,b=25), height=h)
    return fig

def dim_bar(label, score, delay=0):
    c = _color(score)
    st.markdown(f"""<div class="pbar"><div class="pbar-row">
        <span class="pbar-name">{label}</span><span class="pbar-val" style="color:{c}">{score}</span>
    </div><div class="pbar-track"><div class="pbar-fill" style="width:{score}%;background:{c};animation-delay:{delay}s"></div></div></div>""", unsafe_allow_html=True)

def score_ring(score):
    c = _color(score)
    st.markdown(f"""<div class="score-ring" style="background:conic-gradient({c} {score*3.6}deg, rgba(30,42,69,.6) 0deg)">
        <div class="score-ring-inner"><span class="score-ring-val" style="color:{c}">{score}</span>
        <span class="score-ring-sub">of 100</span></div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════
def _build_preview_html():
    dims = demodata.DEMO_SCORES["dimensions"]
    score = demodata.DEMO_SCORES["overall_score"]
    sc = _color(score)
    dim_bars = ""
    for name, val in list(dims.items())[:4]:
        c = _color(val)
        short = name.replace("Technical Skill Match","Tech Match").replace("Communication Clarity","Comms").replace("Evidence Quality","Evidence").replace("Learning Momentum","Momentum")
        dim_bars += f'<div class="preview-dim"><span class="preview-dim-label">{short}</span><div class="preview-dim-track"><div class="preview-dim-fill" style="width:{val}%;background:{c}"></div></div></div>'
    matched = ["Python","React","Docker","Git"]
    gaps = ["Java","PostgreSQL","K8s"]
    m_chips = "".join(f'<span class="preview-chip" style="background:#34d399;color:#1a1a1a">{s}</span>' for s in matched)
    g_chips = "".join(f'<span class="preview-chip" style="background:#f87171">{s}</span>' for s in gaps)
    return f'''<div class="preview-frame">
        <div class="preview-bar"><div class="preview-dot"></div><div class="preview-dot"></div><div class="preview-dot"></div>
            <span style="margin-left:.5rem;font-size:.6rem;color:var(--t4)">TRAXR — Analysis Results</span></div>
        <div class="preview-body">
            <div class="preview-score">
                <div class="preview-ring" style="background:conic-gradient({sc} {score*3.6}deg, rgba(30,42,69,.6) 0deg)">
                    <div class="preview-ring-inner" style="color:{sc}">{score}</div></div>
                <div class="preview-dims">{dim_bars}</div></div>
            <div style="display:flex;gap:var(--s4);margin-top:var(--s3)">
                <div style="flex:1"><div style="font-size:.6rem;color:var(--t4);font-weight:600;margin-bottom:3px">MATCHED</div>{m_chips}</div>
                <div style="flex:1"><div style="font-size:.6rem;color:var(--t4);font-weight:600;margin-bottom:3px">GAPS</div>{g_chips}</div>
            </div></div></div>'''


def render_landing():
    st.markdown('<div class="land-wrap">', unsafe_allow_html=True)
    hero_l, hero_r = st.columns([1.1, 1], gap="large")
    with hero_l:
        st.markdown(f"""<div class="land-hero-left">
            <div class="land-tagline">Track · Prove · Advance</div>
            <div class="t-brand" style="margin-bottom:var(--s5)">TRAXR</div>
            <div class="land-headline">See exactly what skills you're missing for the internship you want.</div>
            <div class="land-desc">Get explainable scoring, proof-first feedback, and a focused 30-day plan to close the gap — before you apply.</div>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="cta-primary">', unsafe_allow_html=True)
            if st.button("▶  Try Demo", key="cta_demo", use_container_width=True):
                st.session_state.view = "demo_generator"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="cta-ghost">', unsafe_allow_html=True)
            if st.button("→  Analyze Now", key="cta_start", use_container_width=True):
                st.session_state.view = "dashboard"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""<div class="land-value">
            <div class="land-value-item"><span>🎯</span> Role-specific</div>
            <div class="land-value-item"><span>📊</span> Explainable</div>
            <div class="land-value-item"><span>🧩</span> Proof-first</div>
        </div>""", unsafe_allow_html=True)
    with hero_r:
        st.markdown(f'<div style="padding:var(--s7) 0 var(--s6)">{_build_preview_html()}</div>', unsafe_allow_html=True)

    # ── Feature Showcase ──
    st.markdown("""<div class="showcase">
        <div class="showcase-eyebrow">How TRAXR Works</div>
        <span class="showcase-headline">A proof-first system for resume readiness</span>
        <div class="showcase-sub">Four pillars that turn vague self-assessment into structured, evidence-backed career clarity.</div>
    </div>""", unsafe_allow_html=True)

    # Row 1
    fc1, fc2 = st.columns(2, gap="medium")
    with fc1:
        st.markdown("""<div class="feature-card">
            <div class="fc-content-top">
                <div class="fc-visual">
                    <div class="fc-icon">🧬</div>
                    <span class="fc-badge fc-badge--req">Required</span>
                    <span class="fc-badge fc-badge--pref">Preferred</span>
                    <span class="fc-badge fc-badge--ctx">Context</span>
                </div>
                <div class="fc-title">Decode the Job Description</div>
                <p class="fc-desc">We extract the real skill tree, behavioral signals, and context behind a role — instead of stopping at raw keyword overlap.</p>
            </div>
            <div class="fc-content-bottom">
                <div class="fc-mini-bars">
                    <div class="fc-mini-bar"><span class="fc-mini-label">Python</span><div class="fc-mini-track"><div class="fc-mini-fill" style="width:92%;background:#34d399"></div></div></div>
                    <div class="fc-mini-bar"><span class="fc-mini-label">System Design</span><div class="fc-mini-track"><div class="fc-mini-fill" style="width:45%;background:#fbbf24"></div></div></div>
                    <div class="fc-mini-bar"><span class="fc-mini-label">K8s</span><div class="fc-mini-track"><div class="fc-mini-fill" style="width:18%;background:#f87171"></div></div></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    with fc2:
        st.markdown("""<div class="feature-card">
            <div class="fc-content-top">
                <div class="fc-visual">
                    <div class="fc-icon">⬡</div>
                </div>
                <div class="fc-title">6-Dimension Scoring</div>
                <p class="fc-desc">TRAXR scores readiness across Technical Match, Evidence Quality, Communication, Alignment, Momentum, and Interview Readiness.</p>
            </div>
            <div class="fc-content-bottom">
                <div class="fc-mini-bars">
                    <div class="fc-mini-bar"><span class="fc-mini-label">Tech Match</span><div class="fc-mini-track"><div class="fc-mini-fill" style="width:72%;background:#818cf8"></div></div></div>
                    <div class="fc-mini-bar"><span class="fc-mini-label">Evidence</span><div class="fc-mini-track"><div class="fc-mini-fill" style="width:85%;background:#818cf8"></div></div></div>
                    <div class="fc-mini-bar"><span class="fc-mini-label">Alignment</span><div class="fc-mini-track"><div class="fc-mini-fill" style="width:54%;background:#818cf8"></div></div></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # MANDATORY ROW SPACER - DO NOT MOVE INSIDE A COLUMN
    st.markdown(
        "<div style='height: 2.5rem; width: 100%; clear: both;'></div>",
        unsafe_allow_html=True
    )

    # Row 2
    fc3, fc4 = st.columns(2, gap="medium")
    with fc3:
        st.markdown("""<div class="feature-card">
            <div class="fc-content-top">
                <div class="fc-visual">
                    <div class="fc-icon">⚡</div>
                </div>
                <div class="fc-title">Prove Your Claims</div>
                <p class="fc-desc">We turn claimed skills into targeted proof tasks so your resume is backed by evidence, not just self-reporting.</p>
            </div>
            <div class="fc-content-bottom">
                <div class="fc-terminal">
                    <span class="t-cmt"># SkillPrint challenge</span><br>
                    <span class="t-kw">def</span> <span class="t-fn">validate_skill</span>(claim):<br>
                    &nbsp;&nbsp;<span class="t-kw">return</span> run_proof(<span class="t-str">"ci/cd"</span>)
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    with fc4:
        st.markdown("""<div class="feature-card">
            <div class="fc-content-top">
                <div class="fc-visual">
                    <div class="fc-icon">🗺️</div>
                </div>
                <div class="fc-title">30-Day Action Plan</div>
                <p class="fc-desc">Get a personalized weekly roadmap tied to your actual gaps, with milestones designed to build visible proof.</p>
            </div>
            <div class="fc-content-bottom">
                <div class="fc-timeline">
                    <div class="fc-tl-bar" style="height:20px"></div>
                    <div class="fc-tl-bar" style="height:30px"></div>
                    <div class="fc-tl-bar" style="height:26px"></div>
                    <div class="fc-tl-bar" style="height:38px"></div>
                    <div class="fc-tl-bar" style="height:32px"></div>
                    <div class="fc-tl-bar" style="height:42px"></div>
                    <div class="fc-tl-bar" style="height:36px"></div>
                </div>
                <div class="fc-tl-labels"><span class="fc-tl-label">Week 1</span><span class="fc-tl-label">Week 4</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="land-footer">TRAXR v1.0 · Track. Prove. Advance.</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════
def render_dashboard():
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown("""<div class="dash-bar"><div class="dash-bar-left">
            <span class="t-brand-sm">TRAXR</span>
            <span class="dash-bar-tag">Dashboard</span>
        </div></div>""", unsafe_allow_html=True)
    with h2:
        if st.button("← Back", key="back_btn"):
            st.session_state.view = "landing"
            st.rerun()

    r = st.session_state.result

    if r is None:
        _render_inputs()
    else:
        with st.expander("📝 Change inputs", expanded=False):
            _render_inputs()
        _render_results(r)

    st.markdown("""<div style="text-align:center;color:var(--t4);font-size:.72rem;padding:var(--s6) 0 var(--s4)">
        TRAXR v1.0 · Track. Prove. Advance.</div>""", unsafe_allow_html=True)


def _render_inputs():
    # Name input
    name_val = st.text_input("Your name (optional)", value=st.session_state.user_name,
                              placeholder="Enter your name for a personalized experience", key="name_in")
    st.session_state.user_name = name_val

    col_l, col_r = st.columns(2, gap="large")

    # ── RESUME CARD (Left) ──
    with col_l:
        uploaded = st.session_state.get("pdf_up")
        ready_class = "input-card-ready" if uploaded else ""
        st.markdown(f'<div class="input-card-wrap {ready_class}">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="input-card-label"><span>📄</span> Resume</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed", key="pdf_up")
            st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)
            resume_paste = st.text_area("Paste Resume", height=150,
                placeholder="Paste the full text of your resume here…",
                label_visibility="collapsed", key="res_paste")
            st.markdown(
                '<div class="helper-text">※ PDF upload takes priority. Ensure text is selectable if using PDF.</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── JOB DESCRIPTION CARD (Right) ──
    with col_r:
        st.markdown('<div class="input-card-wrap">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="input-card-label"><span>💼</span> Job Description</div>', unsafe_allow_html=True)
            jd_input = st.text_area("Paste Job Description", height=328,
                placeholder="Paste the full job description here…",
                label_visibility="collapsed", key="jd_in")
            if jd_input and len(jd_input) < 100:
                st.warning(f"Job description seems short ({len(jd_input)} chars).")
            st.markdown(
                '<div class="helper-text">※ Paste the full text for better semantic matching and gap detection.</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ANALYZE CTA ROW ──
    st.markdown('<div class="analyze-cta-row">', unsafe_allow_html=True)
    _, bc, _ = st.columns([2, 2, 2])
    with bc:
        st.markdown('<div class="cta-primary">', unsafe_allow_html=True)
        analyze_clicked = st.button("🚀  Analyze Readiness", use_container_width=True, key="analyze_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        resume_text = ""
        used_ocr = False
        extraction_method = "none"
        pages_processed = 0
        chars_extracted = 0
        low_quality = False

        jd = jd_input.strip() if jd_input else ""

        # Priority 1: uploaded PDF → staged extraction pipeline
        if uploaded:
            api_key = _get_api_key()
            uploaded.seek(0)

            with st.spinner("Reading PDF text..."):
                raw_text = extract_pdf_text_basic(uploaded)
                low_quality = is_text_low_quality(raw_text)

            if not low_quality and raw_text.strip():
                resume_text = normalize_resume_text(raw_text)
                extraction_method = "pypdf"
                chars_extracted = len(resume_text)
                st.success("Resume parsed successfully.")
            elif api_key:
                with st.spinner("Text layer weak. Running OCR fallback (this may take up to a minute)..."):
                    uploaded.seek(0)
                    page_images = render_pdf_pages_for_ocr(uploaded)
                    pages_processed = len(page_images)
                    ocr_text = extract_text_with_gemini_from_pages(page_images, api_key)
                    used_ocr = True

                if ocr_text and len(ocr_text.strip()) >= 100:
                    with st.spinner("OCR complete."):
                        resume_text = normalize_resume_text(ocr_text)
                    extraction_method = "gemini_ocr"
                    chars_extracted = len(resume_text)
                    st.success("Resume parsed using OCR fallback.")
                elif raw_text.strip():
                    resume_text = normalize_resume_text(raw_text)
                    extraction_method = "pypdf"
                    chars_extracted = len(resume_text)
                    st.warning("We extracted partial text. Review it before analyzing.")
                else:
                    st.error(
                        "OCR could not read this PDF (the API may be rate-limited). "
                        "Please **paste your resume text** in the text box below, "
                        "or try again in a minute."
                    )
            elif raw_text.strip():
                resume_text = normalize_resume_text(raw_text)
                extraction_method = "pypdf"
                chars_extracted = len(resume_text)
                st.warning("We extracted partial text. Review it before analyzing.")
            else:
                st.error("Could not extract readable text from this PDF.")

            # Debug expander (shown when extraction produced something)
            if resume_text:
                with st.expander("🔬 Extraction details", expanded=False):
                    st.caption(f"Method: {extraction_method}")
                    st.caption(f"Characters extracted: {chars_extracted:,}")
                    st.caption(f"Pages processed: {pages_processed}")
                    st.caption(f"OCR used: {'Yes' if used_ocr else 'No'}")
                    st.caption(f"Quality gate: {'Failed (triggered OCR)' if low_quality else 'Passed'}")

        # Priority 2: pasted text
        if not resume_text and resume_paste and resume_paste.strip():
            resume_text = resume_paste.strip()

        # ── Validate ──
        if not resume_text and not jd:
            st.error("Please provide both a **resume** and a **job description** to begin.")
        elif not resume_text:
            st.error("Resume content is missing. Please upload a PDF or paste text.")
        elif not jd:
            st.error("Job description is missing. Please paste the target role description.")
        else:
            _run_analysis(resume_text, jd, False)

    if st.session_state.result is None:
        st.markdown("""<div style="text-align:center;padding:var(--s7) 0;color:var(--t4)">
            <div style="font-size:2rem;margin-bottom:var(--s2);opacity:.5">📊</div>
            <div class="t-body" style="color:var(--t3)">Upload your resume and paste a job description to begin</div>
            <div class="t-caption" style="margin-top:var(--s2)">Or go back and click <b>Try Demo</b> to explore with sample data</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════
def _run_analysis(resume_text: str, jd_text: str, is_demo: bool):
    input_hash = _h(resume_text + jd_text)

    # Skip if same inputs already analyzed
    if (st.session_state.result
            and st.session_state.result.input_hash == input_hash):
        return

    gemini_flags: list[bool] = []
    fallback_reasons: list[str] = []

    # 1. Role DNA
    with st.spinner("🔍 Extracting Role DNA…"):
        dna_raw = build_role_dna(resume_text, jd_text)
        gemini_flags.append(dna_raw.get("_gemini_used", False))
        if dna_raw.get("_fallback_reason"):
            fallback_reasons.append(f"role_dna: {dna_raw['_fallback_reason']}")
        role_dna_obj = RoleDNA.from_dict(dna_raw)
        role_dna_dict = role_dna_obj.to_dict()

    # 2. Profile signals + deterministic scores
    with st.spinner("📊 Analyzing profile…"):
        signals = extract_profile_signals(resume_text)
        resume_skills = extract_skills(resume_text)
        scores_raw = compute_360_score(resume_text, role_dna_dict, signals)

    # 3. Try Gemini-enhanced score breakdown
    with st.spinner("🧠 Generating insights…"):
        score_result = build_score_breakdown(resume_text, jd_text, role_dna_dict)
        gemini_flags.append(score_result.get("_gemini_used", False))
        if score_result.get("_fallback_reason"):
            fallback_reasons.append(f"scores: {score_result['_fallback_reason']}")
        # Use Gemini scores if available, otherwise deterministic
        if score_result.get("_gemini_used"):
            dim_list = score_result.get("dimensions", [])
            overall = score_result.get("overall_score", 0)
            confidence = score_result.get("confidence", "medium")
        else:
            dim_list = scores_raw.get("dimensions", [])
            overall = scores_raw.get("overall_score", 0)
            confidence = scores_raw.get("confidence", "medium")

    score_dims = [ScoreDimension.from_dict(d) for d in dim_list]

    # 4. Gap classification
    with st.spinner("🔎 Classifying gaps…"):
        gaps_result = build_gaps(resume_text, jd_text, role_dna_dict)
        gemini_flags.append(gaps_result.get("_gemini_used", False))
        if gaps_result.get("_fallback_reason"):
            fallback_reasons.append(f"gaps: {gaps_result['_fallback_reason']}")
        # Use Gemini gaps if available, otherwise deterministic
        if gaps_result.get("_gemini_used"):
            gap_dicts = gaps_result.get("items", [])
        else:
            gap_dicts = classify_gaps(resume_text, resume_skills, role_dna_dict)
    gap_items = [GapItem.from_dict(g) for g in gap_dicts]

    # 5. SkillPrint challenge
    with st.spinner("🧩 Creating SkillPrint…"):
        sp_result = build_skillprint(resume_text, jd_text, role_dna_dict, gap_dicts)
        gemini_flags.append(sp_result.get("_gemini_used", False))
        if sp_result.get("_fallback_reason"):
            fallback_reasons.append(f"skillprint: {sp_result['_fallback_reason']}")
    skillprint_obj = SkillPrint.from_dict(sp_result)

    # 6. Roadmap
    with st.spinner("🗺️ Building roadmap…"):
        road_result = build_roadmap(resume_text, jd_text, role_dna_dict, gap_dicts)
        gemini_flags.append(road_result.get("_gemini_used", False))
        if road_result.get("_fallback_reason"):
            fallback_reasons.append(f"roadmap: {road_result['_fallback_reason']}")
    roadmap_items = [RoadmapWeek.from_dict(w) for w in road_result.get("weeks", [])]

    any_gemini = any(gemini_flags)
    any_fallback = bool(fallback_reasons)

    # Build unified result
    from llmclient import _infer_job_title
    st.session_state.result = AnalysisResult(
        user_name=st.session_state.user_name,
        resume_text=resume_text,
        job_description_text=jd_text,
        job_title=_infer_job_title(jd_text),
        role_summary=role_dna_obj.role_context,
        is_demo=is_demo,
        input_hash=input_hash,
        role_dna=role_dna_obj,
        score_breakdown=score_dims,
        overall_score=overall,
        confidence=confidence,
        matched_skills=scores_raw.get("matched_skills", []),
        missing_skills=scores_raw.get("missing_skills", []),
        gaps=gap_items,
        skillprint=skillprint_obj,
        roadmap=roadmap_items,
        gemini_used=any_gemini,
        fallback_used=any_fallback,
        fallback_reasons=fallback_reasons,
    )
    st.session_state.skillprint_code = ""
    st.session_state.skillprint_eval = None
    st.session_state.greeting_shown = False
    st.rerun()


# ═══════════════════════════════════════════════════════════
# RESULTS RENDERING
# ═══════════════════════════════════════════════════════════
def _render_results(r: AnalysisResult):
    # ── Greeting ──
    if st.session_state.user_name:
        name = st.session_state.user_name
        anim_cls = "greeting--static" if st.session_state.greeting_shown else "greeting--anim"
        st.markdown(f'<div class="greeting {anim_cls}">Hey {name}, here\'s your readiness breakdown 👋</div>', unsafe_allow_html=True)
        st.session_state.greeting_shown = True

    # ── Demo + Gemini badges ──
    if r.is_demo:
        st.info("🔔 **Demo mode** — showing sample analysis. Upload your own resume & job description for personalized results.")
    if r.gemini_used:
        badge_line = '<span class="engine-badge engine-badge--gemini">⚡ Powered by Gemini</span>'
    else:
        badge_line = '<span class="engine-badge engine-badge--local">🔧 Local analysis engine</span>'
    st.markdown(f'<div style="margin-bottom:var(--s4)">{badge_line}</div>', unsafe_allow_html=True)
    if r.fallback_used and not r.gemini_used and not r.is_demo:
        st.warning("⚠️ Gemini API was rate-limited. Analysis was generated using the local scoring engine. "
                   "Results are still accurate but less detailed. Try again later for AI-enhanced insights.")

    # ══════ Score Hero ══════
    dims = r.dimensions_dict
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.markdown('<div class="card-title" style="justify-content:center;margin-top:var(--s3)">360° Readiness Score</div>', unsafe_allow_html=True)
        score_ring(r.overall_score)
        conf = r.confidence
        st.markdown(f'<div class="t-caption" style="text-align:center;margin-bottom:var(--s5)">Confidence: <b style="color:var(--t2)">{conf.title() if isinstance(conf,str) else conf}</b></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Dimension Breakdown</div>', unsafe_allow_html=True)
        for i, (d, s) in enumerate(dims.items()):
            dim_bar(d, s, delay=.15 + i * .08)
    with right_col:
        st.markdown('<div class="card-title" style="margin-top:var(--s3)">Readiness Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(build_radar(dims, h=420), use_container_width=True)

    # ══════ Tabs ══════
    st.markdown('<div style="margin-top:var(--s5)"></div>', unsafe_allow_html=True)
    t_ov, t_dna, t_bd, t_gap, t_sp, t_road = st.tabs(
        ["Overview", "Role DNA", "Score Breakdown", "Top Gaps", "SkillPrint", "Roadmap"])

    with t_ov: _tab_overview(r)
    with t_dna: _tab_role_dna(r)
    with t_bd: _tab_score_breakdown(r)
    with t_gap: _tab_top_gaps(r)
    with t_sp: _tab_skillprint(r)
    with t_road: _tab_roadmap(r)


# ── Tab: Overview ──
def _tab_overview(r: AnalysisResult):
    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Score", f"{r.overall_score}/100")
    m2.metric("Matched", len(r.matched_skills))
    m3.metric("Gaps", len(r.gaps))
    m4.metric("Confidence", r.confidence.title() if isinstance(r.confidence, str) else "—")

    mc, gc = st.columns(2, gap="medium")
    matched_html = "".join(f'<span class="chip chip-matched">{s.title()}</span>' for s in r.matched_skills[:12])
    gap_css = {"hard_gap": "hard", "soft_gap": "soft", "context_gap": "context", "proof_gap": "soft"}
    gap_html = "".join(f'<span class="chip chip-{gap_css.get(g.gap_type, "hard")}">{g.name.title()}</span>' for g in r.gaps[:8])
    with mc:
        st.markdown(f'<div class="card"><div class="card-title">✅ Matched Skills</div>{matched_html or "<span class=t-caption>None detected</span>"}</div>', unsafe_allow_html=True)
    with gc:
        st.markdown(f'<div class="card"><div class="card-title">⚠️ Skill Gaps</div>{gap_html or "<span style=color:var(--ok)>No gaps!</span>"}</div>', unsafe_allow_html=True)


# ── Tab: Role DNA ──
def _tab_role_dna(r: AnalysisResult):
    dna = r.role_dna

    # ── Role Context banner (full-width) ──
    st.markdown(f'''<div style="background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(124,58,237,.05));
        border:1px solid rgba(99,102,241,.15);border-radius:var(--r);padding:var(--s5) var(--s6);
        margin-bottom:var(--s5);animation:fadeUp .4s ease both">
        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:var(--s3)">
            <span style="font-size:1.1rem">🎯</span>
            <span style="font-size:.72rem;font-weight:700;color:var(--acc2);text-transform:uppercase;
                letter-spacing:1.5px">Role Context</span>
        </div>
        <p style="font-size:.92rem;color:var(--t1);line-height:1.7;margin:0;font-weight:400">
            {dna.role_context or "No role context available."}</p>
    </div>''', unsafe_allow_html=True)

    # ── Skills section (3-column) ──
    sc1, sc2, sc3 = st.columns(3, gap="medium")

    # Required Skills
    with sc1:
        req_chips = "".join(
            f'<span style="display:inline-block;padding:5px 12px;margin:3px;border-radius:8px;'
            f'font-size:.76rem;font-weight:600;color:#f87171;'
            f'background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.18)">{s.title()}</span>'
            for s in dna.required_skills
        ) if dna.required_skills else '<span style="color:var(--t4);font-size:.85rem">—</span>'
        st.markdown(f'''<div style="background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);
            padding:var(--s5);height:100%;animation:fadeUp .45s ease .05s both">
            <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:var(--s4)">
                <div style="width:8px;height:8px;border-radius:50%;background:#f87171"></div>
                <span style="font-size:.72rem;font-weight:700;color:var(--t3);text-transform:uppercase;
                    letter-spacing:.8px">Required Skills</span>
                <span style="margin-left:auto;font-size:.65rem;font-weight:700;color:var(--t4);
                    background:rgba(248,113,113,.08);padding:1px 8px;border-radius:10px">{len(dna.required_skills)}</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:2px">{req_chips}</div>
        </div>''', unsafe_allow_html=True)

    # Preferred Skills
    with sc2:
        pref_chips = "".join(
            f'<span style="display:inline-block;padding:5px 12px;margin:3px;border-radius:8px;'
            f'font-size:.76rem;font-weight:600;color:#fbbf24;'
            f'background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.15)">{s.title()}</span>'
            for s in dna.preferred_skills
        ) if dna.preferred_skills else '<span style="color:var(--t4);font-size:.85rem">—</span>'
        st.markdown(f'''<div style="background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);
            padding:var(--s5);height:100%;animation:fadeUp .45s ease .1s both">
            <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:var(--s4)">
                <div style="width:8px;height:8px;border-radius:50%;background:#fbbf24"></div>
                <span style="font-size:.72rem;font-weight:700;color:var(--t3);text-transform:uppercase;
                    letter-spacing:.8px">Preferred Skills</span>
                <span style="margin-left:auto;font-size:.65rem;font-weight:700;color:var(--t4);
                    background:rgba(251,191,36,.06);padding:1px 8px;border-radius:10px">{len(dna.preferred_skills)}</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:2px">{pref_chips}</div>
        </div>''', unsafe_allow_html=True)

    # Tools
    with sc3:
        tool_chips = "".join(
            f'<span style="display:inline-block;padding:5px 12px;margin:3px;border-radius:8px;'
            f'font-size:.76rem;font-weight:600;color:var(--acc3);'
            f'background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.12)">{s.title()}</span>'
            for s in dna.tools
        ) if dna.tools else '<span style="color:var(--t4);font-size:.85rem">—</span>'
        st.markdown(f'''<div style="background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);
            padding:var(--s5);height:100%;animation:fadeUp .45s ease .15s both">
            <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:var(--s4)">
                <span style="font-size:.7rem">🛠️</span>
                <span style="font-size:.72rem;font-weight:700;color:var(--t3);text-transform:uppercase;
                    letter-spacing:.8px">Tools & Platforms</span>
                <span style="margin-left:auto;font-size:.65rem;font-weight:700;color:var(--t4);
                    background:rgba(99,102,241,.06);padding:1px 8px;border-radius:10px">{len(dna.tools)}</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:2px">{tool_chips}</div>
        </div>''', unsafe_allow_html=True)

    # ── Bottom row: Day-to-Day + Behavioral ──
    b1, b2 = st.columns([3, 2], gap="medium")

    with b1:
        acts_html = ""
        for i, a in enumerate(dna.day_to_day):
            acts_html += (
                f'<div style="display:flex;align-items:flex-start;gap:.7rem;padding:.55rem 0;'
                f'border-bottom:1px solid rgba(148,163,184,.05)">'
                f'<span style="flex-shrink:0;width:20px;height:20px;border-radius:6px;'
                f'background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.12);'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:.6rem;font-weight:700;color:var(--acc2);margin-top:1px">{i+1}</span>'
                f'<span style="font-size:.85rem;color:var(--t2);line-height:1.5">{a}</span>'
                f'</div>'
            )
        if not acts_html:
            acts_html = '<span style="color:var(--t4);font-size:.85rem">—</span>'
        st.markdown(f'''<div style="background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);
            padding:var(--s5);margin-top:var(--s4);animation:fadeUp .45s ease .2s both">
            <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:var(--s3)">
                <span style="font-size:.8rem">💼</span>
                <span style="font-size:.72rem;font-weight:700;color:var(--t3);text-transform:uppercase;
                    letter-spacing:.8px">Day-to-Day Responsibilities</span>
            </div>
            {acts_html}
        </div>''', unsafe_allow_html=True)

    with b2:
        behav_chips = "".join(
            f'<span style="display:inline-block;padding:6px 14px;margin:3px;border-radius:20px;'
            f'font-size:.78rem;font-weight:500;color:var(--t2);'
            f'background:rgba(148,163,184,.06);border:1px solid rgba(148,163,184,.1)">{t.title()}</span>'
            for t in dna.behavioral_traits
        ) if dna.behavioral_traits else '<span style="color:var(--t4);font-size:.85rem">—</span>'
        st.markdown(f'''<div style="background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);
            padding:var(--s5);margin-top:var(--s4);animation:fadeUp .45s ease .25s both">
            <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:var(--s4)">
                <span style="font-size:.8rem">🧠</span>
                <span style="font-size:.72rem;font-weight:700;color:var(--t3);text-transform:uppercase;
                    letter-spacing:.8px">Behavioral Traits</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:2px">{behav_chips}</div>
        </div>''', unsafe_allow_html=True)


# ── Tab: Score Breakdown ──
def _tab_score_breakdown(r: AnalysisResult):
    for dim in r.score_breakdown:
        emoji = "🟢" if dim.score >= 75 else "🟡" if dim.score >= 50 else "🔴"
        with st.expander(f"{emoji}  {dim.label} — {dim.score}/100", expanded=False):
            st.markdown(f"**Why this score?** {dim.why or '—'}")
            if dim.evidence_found:
                st.markdown("**Evidence found:** " + " · ".join(str(x) for x in dim.evidence_found[:6]))
            if dim.missing_proof:
                st.markdown("**Missing:** " + " · ".join(str(x) for x in dim.missing_proof[:5]))
            st.markdown(f"**Recommended action:** {dim.action or '—'}")


# ── Tab: Top Gaps ──
def _tab_top_gaps(r: AnalysisResult):
    if not r.gaps:
        st.success("🎉 No significant gaps — you're a strong fit!")
        return
    for g in r.top_gaps:
        sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(g.severity, "⚪")
        type_label = {"hard_gap": "Hard Gap", "soft_gap": "Soft Gap", "context_gap": "Context Gap", "proof_gap": "Proof Gap"}.get(g.gap_type, "Gap")
        with st.expander(f'{sev_icon}  {g.name.title()} — {type_label} ({g.severity})', expanded=(g.severity == "high")):
            st.markdown(f'**Category:** {g.category or "—"}')
            st.markdown(f'**Why it matters:** {g.why_it_matters}')
            if g.evidence_missing:
                st.markdown(f'**What\'s missing from your resume:** {g.evidence_missing}')
            st.markdown(f'**Action:** {g.action}')


# ── Tab: SkillPrint ──
def _tab_skillprint(r: AnalysisResult):
    sp = r.skillprint
    if not sp or not sp.title:
        st.info("No SkillPrint challenge — all critical skills are matched!")
        return

    st.markdown(f'<div class="card"><div class="card-title">🧩 {sp.title}</div>', unsafe_allow_html=True)
    st.markdown(sp.challenge_brief)
    if sp.proof_signal:
        st.markdown(f'**Resume impact:** {sp.proof_signal}')
    if sp.deliverables:
        st.markdown("**Deliverables:**")
        for d in sp.deliverables:
            st.markdown(f"- {d}")
    if sp.rubric:
        st.markdown("**Rubric:**")
        for rb in sp.rubric:
            st.markdown(f"- {rb}")
    st.markdown('</div>', unsafe_allow_html=True)

    code = st.text_area("✍️ Your solution", value=st.session_state.skillprint_code or sp.starter_prompt, height=200, key="sp_code")
    st.session_state.skillprint_code = code

    if st.button("🚀 Submit", key="sp_sub"):
        with st.spinner("Evaluating…"):
            result = evaluate_skillprint_submission(code, sp.to_dict())
            st.session_state.skillprint_eval = result
            if result.get("passed") and st.session_state.result:
                # Boost technical match score
                new_dims = []
                for d in r.score_breakdown:
                    if d.key == "technical_match":
                        new_dims.append(ScoreDimension(
                            key=d.key, label=d.label, score=min(100, d.score + 8),
                            why=d.why, evidence_found=d.evidence_found,
                            missing_proof=d.missing_proof, action=d.action,
                        ))
                    else:
                        new_dims.append(d)
                new_overall = round(sum(d.score for d in new_dims) / max(1, len(new_dims)))
                st.session_state.result = AnalysisResult(
                    user_name=r.user_name, resume_text=r.resume_text,
                    job_description_text=r.job_description_text, job_title=r.job_title,
                    role_summary=r.role_summary, is_demo=r.is_demo, input_hash=r.input_hash,
                    role_dna=r.role_dna, score_breakdown=new_dims, overall_score=new_overall,
                    confidence=r.confidence, matched_skills=r.matched_skills,
                    missing_skills=r.missing_skills, gaps=r.gaps, skillprint=r.skillprint,
                    roadmap=r.roadmap, gemini_used=r.gemini_used,
                    fallback_used=r.fallback_used, fallback_reasons=r.fallback_reasons,
                )
                st.rerun()

    ev = st.session_state.skillprint_eval
    if ev:
        rc = _color(ev.get("score", 0))
        st.markdown(f'<div class="card"><div class="card-title" style="color:{rc}">Score: {ev.get("score", 0)}/100 {"✅ Passed!" if ev.get("passed") else "❌ Not yet"}</div>', unsafe_allow_html=True)
        st.markdown(f'**Feedback:** {ev.get("feedback", "—")}')
        for imp in ev.get("improvements", []):
            st.markdown(f"- {imp}")
        st.markdown('</div>', unsafe_allow_html=True)


# ── Tab: Roadmap ──
def _tab_roadmap(r: AnalysisResult):
    if not r.roadmap:
        st.info("Roadmap will appear after analysis.")
        return
    for week in r.roadmap:
        focus = ", ".join(s.title() for s in week.focus_skills) if week.focus_skills else ""
        st.markdown(f"### 📅 Week {week.week} — Focus: {focus}")
        if week.tasks:
            tasks_html = "".join(f"<li>{t}</li>" for t in week.tasks)
            st.markdown(f'<div class="road-item"><h4>Day {(week.week - 1) * 7 + 1}-{week.week * 7} — {week.goal}</h4><ul>{tasks_html}</ul></div>', unsafe_allow_html=True)
        for res in week.resources:
            if isinstance(res, dict):
                st.markdown(f'📎 [{res.get("name", "")}]({res.get("url", "")})')
            else:
                st.markdown(f"📎 {res}")
        if week.proof_milestone:
            st.caption(f"✅ Proof: {week.proof_milestone}")


# ═══════════════════════════════════════════════════════════
# DEMO GENERATOR PAGE
# ═══════════════════════════════════════════════════════════

_DEMO_RESUME_PROMPT = """Generate a realistic, plain-text resume for a random tech professional.
Pick ONE of these roles randomly: Frontend Developer, Backend Developer, Full-Stack Engineer,
Data Scientist, ML Engineer, DevOps Engineer, Mobile Developer, Cloud Architect.

The resume MUST include:
- Name (make one up), contact info, location
- A 2-3 sentence professional summary
- 2-3 work experiences with company names, dates, bullet points showing accomplishments
- Education section
- Technical skills section
- Optionally: projects, certifications

IMPORTANT RULES:
- Output ONLY the resume text, no markdown formatting, no code fences
- Use plain text with simple line breaks and bullet points (- or •)
- Make the candidate good but imperfect — include some skill gaps
- Make experiences realistic with metrics where possible
- Keep it 300-500 words"""

_DEMO_JD_PROMPT = """Generate a realistic job description for a random tech role at a fictional company.
Pick ONE of these roles randomly: Software Engineer Intern, Frontend Developer, Backend Engineer,
Full-Stack Developer, Data Analyst, ML Engineer, DevOps Engineer, Platform Engineer.

The JD MUST include:
- Job title and company name (make one up)
- About the company (2-3 sentences)
- Role description (2-3 sentences)
- Required qualifications (5-8 bullet points)
- Preferred qualifications (3-5 bullet points)
- Responsibilities (5-7 bullet points)

IMPORTANT RULES:
- Output ONLY the job description text, no markdown formatting, no code fences
- Use plain text with simple line breaks and bullet points (- or •)
- Make requirements realistic — mix of must-haves and stretch goals
- Include specific technologies, tools, and soft skills
- Keep it 250-450 words"""


def _generate_with_gemini(prompt: str) -> str:
    """Generate text using Gemini with key rotation."""
    from key_pool import get_all_api_keys, mark_key_exhausted
    import time

    keys = get_all_api_keys()
    if not keys:
        return ""

    for key in keys:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=1.0),
            )
            text = (response.text or "").strip()
            # Strip markdown fencing if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            return text.strip()
        except Exception as exc:
            if "429" in str(exc):
                mark_key_exhausted(key)
                continue
            return ""
    return ""


def render_demo_page():
    """Render the Interactive Demo Generator page."""

    # ── Header bar ──
    hdr_l, hdr_r = st.columns([4, 1])
    with hdr_l:
        st.markdown(f'''<div class="dash-bar-left" style="gap:.6rem">
            <span class="t-brand-sm">TRAXR</span>
            <span class="dash-bar-tag" style="background:rgba(251,191,36,.08);color:#fbbf24;
                border-color:rgba(251,191,36,.2)">🧪 Demo Lab</span>
        </div>''', unsafe_allow_html=True)
    with hdr_r:
        st.markdown('<div class="cta-ghost">', unsafe_allow_html=True)
        if st.button("← Back", key="demo_back", use_container_width=True):
            st.session_state.view = "landing"
            st.session_state.demo_generated_resume = ""
            st.session_state.demo_generated_jd = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr style="margin:.5rem 0 1rem">', unsafe_allow_html=True)

    # ── Compact Premium Banner ──
    st.markdown('''<div class="demo-banner">
        <div class="demo-banner-row">
            <span class="demo-banner-icon">🧪</span>
            <span class="demo-banner-title">Interactive Demo Mode</span>
        </div>
        <div class="demo-banner-sub">
            Generate sample inputs, edit them if needed, then analyze the fit.
        </div>
        <div class="demo-flow-steps">
            <div class="demo-flow-step"><span>1</span> Generate</div>
            <div class="demo-flow-arrow">▸</div>
            <div class="demo-flow-step"><span>2</span> Edit</div>
            <div class="demo-flow-arrow">▸</div>
            <div class="demo-flow-step"><span>3</span> Analyze</div>
        </div>
    </div>''', unsafe_allow_html=True)

    # ── Generate Both button ──
    _, gen_both_col, _ = st.columns([1, 2, 1])
    with gen_both_col:
        st.markdown('<div class="demo-gen-cta">', unsafe_allow_html=True)
        if st.button("✨  Generate Demo Inputs", key="gen_both", use_container_width=True, type="tertiary"):
            with st.spinner("Generating resume and job description..."):
                r_text = _generate_with_gemini(_DEMO_RESUME_PROMPT)
                j_text = _generate_with_gemini(_DEMO_JD_PROMPT)
                if r_text:
                    st.session_state.demo_generated_resume = r_text
                if j_text:
                    st.session_state.demo_generated_jd = j_text
                if r_text or j_text:
                    st.rerun()
                else:
                    st.error("Generation failed — API may be rate-limited. Try again in a moment.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Two-column generator ──
    col_r, col_j = st.columns(2, gap="large")

    with col_r:
        st.markdown(f'''<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:var(--s3)">
            <span style="font-size:1rem">👤</span>
            <span style="font-size:.75rem;font-weight:700;color:var(--t3);text-transform:uppercase;
                letter-spacing:1px">Candidate Resume</span>
        </div>''', unsafe_allow_html=True)

        demo_resume = st.text_area(
            "Resume",
            value=st.session_state.demo_generated_resume,
            height=340,
            placeholder="Click 'Generate' below to create a random candidate profile, or paste your own...",
            label_visibility="collapsed",
            key="demo_resume_area",
        )
        # Sync edits back
        if demo_resume != st.session_state.demo_generated_resume:
            st.session_state.demo_generated_resume = demo_resume

        st.markdown('<div class="demo-gen-sm" style="margin-top:var(--s2)">', unsafe_allow_html=True)
        if st.button("✨  Generate Resume", key="gen_resume", use_container_width=True, type="tertiary"):
            with st.spinner("Generating candidate profile..."):
                text = _generate_with_gemini(_DEMO_RESUME_PROMPT)
                if text:
                    st.session_state.demo_generated_resume = text
                    st.rerun()
                else:
                    st.error("Generation failed — API may be rate-limited. Try again in a moment.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_j:
        st.markdown(f'''<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:var(--s3)">
            <span style="font-size:1rem">💼</span>
            <span style="font-size:.75rem;font-weight:700;color:var(--t3);text-transform:uppercase;
                letter-spacing:1px">Job Description</span>
        </div>''', unsafe_allow_html=True)

        demo_jd = st.text_area(
            "Job Description",
            value=st.session_state.demo_generated_jd,
            height=340,
            placeholder="Click 'Generate' below to create a random job role, or paste your own...",
            label_visibility="collapsed",
            key="demo_jd_area",
        )
        if demo_jd != st.session_state.demo_generated_jd:
            st.session_state.demo_generated_jd = demo_jd

        st.markdown('<div class="demo-gen-sm" style="margin-top:var(--s2)">', unsafe_allow_html=True)
        if st.button("✨  Generate Job Description", key="gen_jd", use_container_width=True, type="tertiary"):
            with st.spinner("Generating job description..."):
                text = _generate_with_gemini(_DEMO_JD_PROMPT)
                if text:
                    st.session_state.demo_generated_jd = text
                    st.rerun()
                else:
                    st.error("Generation failed — API may be rate-limited. Try again in a moment.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Status indicators ──
    has_resume = bool(st.session_state.demo_generated_resume.strip())
    has_jd = bool(st.session_state.demo_generated_jd.strip())

    r_icon = "✅" if has_resume else "⬜"
    j_icon = "✅" if has_jd else "⬜"
    st.markdown(f'''<div style="display:flex;justify-content:center;gap:var(--s6);margin:var(--s5) 0 var(--s3);
        font-size:.78rem;color:var(--t3)">
        <span>{r_icon} Resume ready</span>
        <span>{j_icon} Job description ready</span>
    </div>''', unsafe_allow_html=True)

    # ── Analyze CTA ──
    st.markdown('<div class="analyze-cta-row">', unsafe_allow_html=True)
    _, bc, _ = st.columns([2, 2, 2])
    with bc:
        st.markdown('<div class="cta-primary">', unsafe_allow_html=True)
        if st.button("🚀  Analyze Demo Fit", use_container_width=True, key="demo_analyze_btn"):
            resume = st.session_state.demo_generated_resume.strip()
            jd = st.session_state.demo_generated_jd.strip()
            if not resume or not jd:
                st.error("Please generate (or paste) both a resume and a job description first.")
            else:
                st.session_state.view = "dashboard"
                _run_analysis(resume, jd, is_demo=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer tip ──
    st.markdown(f'''<div style="text-align:center;color:var(--t4);font-size:.72rem;padding:var(--s5) 0 var(--s3);
        border-top:1px solid var(--bdr);margin-top:var(--s6)">
        💡 Tip: You can edit the generated text before analyzing.
        Want to use your real resume? ← Go back and click <b>Analyze Now</b>.
    </div>''', unsafe_allow_html=True)

    # ── Late-injected button overrides ──
    # Generate buttons have class em9zgd03, action buttons have em9zgd02.
    # We target em9zgd03 to style only the generate buttons.
    st.markdown('''<style>
    button.em9zgd03,
    button[class*="em9zgd03"] {
        background: linear-gradient(135deg, #10152a 0%, #141b30 100%) !important;
        border: 1px solid rgba(99, 102, 241, .3) !important;
        border-radius: 10px !important;
        color: #c4b5fd !important;
        font-weight: 600 !important;
        font-size: .8rem !important;
        padding: .5rem 1.2rem !important;
        letter-spacing: .2px !important;
        transition: all .25s ease !important;
    }
    button.em9zgd03:hover,
    button[class*="em9zgd03"]:hover {
        color: #fff !important;
        background: linear-gradient(135deg, #141b30, #1e293b) !important;
        border-color: rgba(99, 102, 241, .5) !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, .15) !important;
        transform: translateY(-1px) !important;
    }
    button.em9zgd03:active,
    button[class*="em9zgd03"]:active {
        transform: scale(.97) !important;
    }
    </style>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE ROUTER
# ═══════════════════════════════════════════════════════════
if st.session_state.view == "landing":
    render_landing()
elif st.session_state.view == "demo_generator":
    render_demo_page()
else:
    render_dashboard()