# -*- coding: utf-8 -*-
"""
BétonFiabilité — Analyse statistique et probabiliste de la résistance du béton.
Application pédagogique Streamlit. Les résultats ne constituent pas une certification normative.
"""

from __future__ import annotations

import io
import json
import math
import re
import uuid
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats as scipy_stats
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

APP_NAME = "BétonFiabilité"
STORE_DIR = Path.home() / ".betonfiabilite"
STORE_FILE = STORE_DIR / "analyses.json"

# ---------------------------------------------------------------------------
# Calculs scientifiques — fonctions séparées de l'interface
# ---------------------------------------------------------------------------

def validate_resistances(values):
    """Retourne un tableau de résistances strictement positives et finies."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Saisissez au moins une résistance.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Les résistances doivent être des nombres finis.")
    if np.any(arr <= 0):
        raise ValueError("Toutes les résistances doivent être strictement positives.")
    return arr


def descriptive_statistics(values, fck):
    """Statistiques descriptives ; écart-type et variance d'échantillon (ddof=1)."""
    x = validate_resistances(values)
    n = int(x.size)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1)) if n >= 2 else None
    std = float(np.std(x, ddof=1)) if n >= 2 else None
    cv = float(std / mean * 100) if std is not None and mean > 0 else None
    below = int(np.sum(x < float(fck)))
    return {
        "n": n,
        "mean": mean,
        "variance": variance,
        "std": std,
        "cv": cv,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "q1": float(np.quantile(x, 0.25)),
        "q3": float(np.quantile(x, 0.75)),
        "below_count": below,
        "below_fraction": below / n,
    }


def normal_probability(mean, std, threshold):
    """P(X < threshold) pour une loi normale N(mean, std²)."""
    mean, std, threshold = float(mean), float(std), float(threshold)
    if mean <= 0 or threshold <= 0 or std < 0:
        raise ValueError("La moyenne et le seuil doivent être positifs ; l'écart-type ne peut pas être négatif.")
    if std == 0:
        return 1.0 if mean < threshold else 0.0
    return float(scipy_stats.norm.cdf(threshold, loc=mean, scale=std))


def lognormal_parameters_from_mean_std(mean, std):
    """Convertit moyenne/écart-type arithmétiques en paramètres mu_ln et sigma_ln."""
    mean, std = float(mean), float(std)
    if mean <= 0 or std < 0:
        raise ValueError("La moyenne doit être positive et l'écart-type non négatif.")
    if std == 0:
        return math.log(mean), 0.0
    sigma_ln = math.sqrt(math.log1p((std / mean) ** 2))
    mu_ln = math.log(mean) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln


def lognormal_probability(mean, std, threshold):
    """P(X < threshold) pour une lognormale paramétrée par moyenne et écart-type arithmétiques."""
    mean, std, threshold = float(mean), float(std), float(threshold)
    if mean <= 0 or threshold <= 0 or std < 0:
        raise ValueError("La moyenne et le seuil doivent être positifs ; l'écart-type ne peut pas être négatif.")
    if std == 0:
        return 1.0 if mean < threshold else 0.0
    mu_ln, sigma_ln = lognormal_parameters_from_mean_std(mean, std)
    return float(scipy_stats.lognorm.cdf(threshold, s=sigma_ln, scale=math.exp(mu_ln)))


def wilson_interval(successes, total, confidence=0.95):
    """Intervalle de Wilson pour une proportion binomiale."""
    if total <= 0:
        return None, None
    z = float(scipy_stats.norm.ppf(0.5 + confidence / 2))
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) / total) + z * z / (4 * total * total)) / den
    return max(0.0, center - half), min(1.0, center + half)


def empirical_probability(values, threshold):
    x = validate_resistances(values)
    count = int(np.sum(x < float(threshold)))
    p = count / len(x)
    low, high = wilson_interval(count, len(x))
    return {"count": count, "n": len(x), "probability": p, "ci_low": low, "ci_high": high}


def simulate_resistances(mean, std, threshold, distribution, n_simulations, seed):
    """Simulation reproductible ; aucune troncature silencieuse de la normale."""
    mean, std, threshold = float(mean), float(std), float(threshold)
    if mean <= 0 or threshold <= 0 or std < 0:
        raise ValueError("Paramètres invalides : moyenne/seuil positifs et écart-type non négatif requis.")
    if n_simulations < 1:
        raise ValueError("Le nombre de simulations doit être supérieur ou égal à 1.")
    rng = np.random.default_rng(int(seed))
    if std == 0:
        samples = np.full(int(n_simulations), mean, dtype=float)
    elif distribution == "Normale":
        samples = rng.normal(mean, std, int(n_simulations))
    elif distribution == "Lognormale":
        mu_ln, sigma_ln = lognormal_parameters_from_mean_std(mean, std)
        samples = rng.lognormal(mu_ln, sigma_ln, int(n_simulations))
    else:
        raise ValueError("Distribution non prise en charge.")
    count = int(np.sum(samples < threshold))
    p = count / int(n_simulations)
    se = math.sqrt(p * (1 - p) / int(n_simulations))
    ci_low, ci_high = wilson_interval(count, int(n_simulations))
    return {
        "samples": samples,
        "count": count,
        "n": int(n_simulations),
        "probability": p,
        "standard_error": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "negative_fraction": float(np.mean(samples < 0)),
    }


def compressive_strength(load_kn, geometry, dimension_mm, second_dimension_mm=None):
    """Résistance en MPa : 1 MPa = 1 N/mm² ; charge d'entrée en kN."""
    load_kn = float(load_kn)
    d1 = float(dimension_mm)
    if load_kn <= 0 or d1 <= 0:
        raise ValueError("La charge et les dimensions doivent être positives.")
    if geometry == "Cylindre":
        area = math.pi * d1 ** 2 / 4
        formula = "A = π d² / 4"
        dimensions = f"diamètre = {d1:g} mm"
    elif geometry == "Cube":
        area = d1 ** 2
        formula = "A = a²"
        dimensions = f"côté = {d1:g} mm"
    else:
        raise ValueError("Géométrie inconnue.")
    load_n = load_kn * 1000
    strength = load_n / area
    return {
        "area_mm2": area,
        "load_n": load_n,
        "strength_mpa": strength,
        "area_formula": formula,
        "dimensions": dimensions,
    }


def distribution_values(mean, std, threshold, distribution, points=500):
    """Prépare densité et fonction de répartition pour le graphique."""
    if std == 0:
        x = np.linspace(max(0.001, min(mean, threshold) * 0.5), max(mean, threshold) * 1.5 + 1, points)
        return x, np.zeros_like(x), np.where(x >= mean, 1.0, 0.0)
    lower = max(0.001, min(mean - 4 * std, threshold - 2 * std))
    upper = max(mean + 4 * std, threshold + 2 * std, lower + 1)
    x = np.linspace(lower, upper, points)
    if distribution == "Normale":
        pdf = scipy_stats.norm.pdf(x, loc=mean, scale=std)
        cdf = scipy_stats.norm.cdf(x, loc=mean, scale=std)
    else:
        mu_ln, sigma_ln = lognormal_parameters_from_mean_std(mean, std)
        pdf = scipy_stats.lognorm.pdf(x, s=sigma_ln, scale=math.exp(mu_ln))
        cdf = scipy_stats.lognorm.cdf(x, s=sigma_ln, scale=math.exp(mu_ln))
    return x, pdf, cdf


def build_pdf_report(project_name, fck, stats_data, probability, model_name, source_label, notes):
    """Construit un rapport PDF en mémoire."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#16324F"), fontSize=20, leading=24))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11,
                              textColor=colors.HexColor("#555555")))
    story = [
        Paragraph("BÉTONFIABILITÉ", styles["ReportTitle"]),
        Paragraph("Rapport d'analyse statistique et probabiliste", styles["Heading2"]),
        Spacer(1, 5*mm),
        Paragraph(f"<b>Projet :</b> {xml_escape(str(project_name))}", styles["BodyText"]),
        Paragraph(f"<b>Date du rapport :</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["BodyText"]),
        Paragraph(f"<b>Source :</b> {xml_escape(str(source_label))}", styles["BodyText"]),
        Paragraph(f"<b>Seuil étudié fck :</b> {fck:.3f} MPa", styles["BodyText"]),
        Spacer(1, 5*mm),
        Paragraph("1. Résultats statistiques", styles["Heading2"]),
    ]
    rows = [["Indicateur", "Valeur"]]
    for label, key, unit in [
        ("Nombre de résultats", "n", ""),
        ("Moyenne", "mean", "MPa"),
        ("Écart-type expérimental", "std", "MPa"),
        ("Variance expérimentale", "variance", "MPa²"),
        ("Coefficient de variation", "cv", "%"),
        ("Minimum", "min", "MPa"),
        ("Maximum", "max", "MPa"),
        ("Médiane", "median", "MPa"),
        ("1er quartile", "q1", "MPa"),
        ("3e quartile", "q3", "MPa"),
    ]:
        value = stats_data.get(key)
        rows.append([label, "Non disponible" if value is None else f"{value:.4g} {unit}".strip()])
    table = Table(rows, colWidths=[90*mm, 65*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16324F")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D6DEE8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F6FA")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.extend([table, Spacer(1, 6*mm), Paragraph("2. Estimation probabiliste", styles["Heading2"])])
    story.append(Paragraph(f"<b>Modèle :</b> {model_name}", styles["BodyText"]))
    story.append(Paragraph(f"<b>P(X &lt; fck) :</b> {probability:.6%}", styles["BodyText"]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("3. Interprétation et limites", styles["Heading2"]))
    story.append(Paragraph(notes.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(
        "Avertissement : ce rapport présente une analyse statistique et probabiliste pédagogique. "
        "Il ne constitue ni une certification de conformité d'un lot, ni une vérification de sécurité structurale. "
        "La validité des conclusions dépend de la représentativité des données, du protocole d'essai et du modèle retenu.",
        styles["SmallNote"]
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Données et persistance locale
# ---------------------------------------------------------------------------

def load_store():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_FILE.exists():
        return []
    try:
        data = json.loads(STORE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def write_store(records):
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = STORE_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(STORE_FILE)


def parse_manual_values(raw):
    """Accepte des nombres séparés par point-virgule, espace ou retour à la ligne."""
    tokens = [t for t in re.split(r"[;\s]+", raw.strip()) if t]
    if not tokens:
        raise ValueError("Aucune valeur n'a été saisie.")
    try:
        return validate_resistances([float(t.replace(",", ".")) for t in tokens])
    except ValueError as exc:
        if "could not convert" in str(exc):
            raise ValueError("Format incorrect. Utilisez par exemple : 28.5; 30.2; 31.0")
        raise


def set_demo_data():
    demo = np.array([
        26.8, 29.4, 30.1, 32.5, 31.7, 28.9, 34.2, 30.8, 33.1, 27.9,
        31.3, 29.8, 35.0, 32.2, 30.5, 28.4, 33.7, 31.0, 29.1, 34.5
    ], dtype=float)
    st.session_state["resistances"] = demo.tolist()
    st.session_state["fck"] = 25.0
    st.session_state["project_name"] = "Exemple pédagogique — Béton fictif"
    st.session_state["lot_name"] = "LOT-DEMO-001"
    st.session_state["age_days"] = 28
    st.session_state["specimen_type"] = "Cylindre"
    st.session_state["data_source"] = "Exemple fictif (données simulées)"
    st.session_state["is_demo"] = True
    sync_widget_defaults()


def init_state():
    defaults = {
        "resistances": [],
        "fck": 25.0,
        "project_name": "Nouvelle analyse",
        "lot_name": "",
        "age_days": 28,
        "specimen_type": "Cylindre",
        "data_source": "Aucune donnée",
        "is_demo": False,
        "analysis_history": load_store(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_widget_to_state(widget_key, data_key):
    """Copie une modification de widget vers les données persistantes de la session."""
    st.session_state[data_key] = st.session_state[widget_key]


def sync_widget_defaults():
    """Réaligne les widgets avec les données après chargement d'un exemple ou d'une analyse."""
    mapping = {
        "fck_dashboard": "fck",
        "fck_data": "fck",
        "fck_probability": "fck",
        "project_name_input": "project_name",
        "lot_name_input": "lot_name",
        "age_days_input": "age_days",
        "specimen_type_input": "specimen_type",
    }
    for widget_key, data_key in mapping.items():
        st.session_state[widget_key] = st.session_state[data_key]


def current_stats():
    if len(st.session_state["resistances"]) < 2:
        return None
    return descriptive_statistics(st.session_state["resistances"], st.session_state["fck"])


def metric_card(label, value, detail="", accent="#16324F"):
    st.markdown(
        f"""<div class="metric-card" style="border-top:4px solid {accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-detail">{detail}</div></div>""",
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(f'<div class="page-kicker">BÉTONFIABILITÉ / {title.upper()}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def get_probability_parameters():
    data = st.session_state["resistances"]
    use_sample = bool(data) and st.radio(
        "Origine des paramètres",
        ["Estimer à partir des essais", "Saisir μ et σ manuellement"],
        horizontal=True,
        key="prob_param_source",
        help="Les paramètres manuels sont utiles pour étudier un scénario théorique."
    ) == "Estimer à partir des essais"
    if use_sample:
        x = validate_resistances(data)
        if len(x) < 2:
            st.warning("Il faut au moins deux résultats pour estimer l'écart-type expérimental.")
            return None
        return float(np.mean(x)), float(np.std(x, ddof=1)), "Paramètres estimés sur les essais"
    col1, col2 = st.columns(2)
    with col1:
        mean = st.number_input("Moyenne μ (MPa)", min_value=0.001, value=30.0, step=0.5, key="manual_mu")
    with col2:
        std = st.number_input("Écart-type σ (MPa)", min_value=0.0, value=3.0, step=0.1, key="manual_sigma")
    return float(mean), float(std), "Paramètres saisis manuellement"


# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(page_title=APP_NAME, page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")
init_state()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#16324F; --blue:#245B83; --orange:#F28C28; --paper:#F3F6FA; --ink:#1D2939; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#F7F9FC 0%,#EEF3F8 100%); color:var(--ink); }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--navy); }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1400px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#102A43,#16324F); }
[data-testid="stSidebar"] * { color:#F5F8FC !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color:#EAF1F8 !important; }
.hero { background:linear-gradient(120deg,#102A43 0%,#245B83 72%,#357A9A 100%); color:white;
        border-radius:22px; padding:28px 32px; margin-bottom:20px; box-shadow:0 12px 28px rgba(16,42,67,.16); }
.hero h1 { color:white !important; margin:0 0 8px 0; font-size:2.2rem; }
.hero p { color:#E4EDF5; font-size:1rem; margin:0; max-width:850px; }
.hero-tag { display:inline-block; background:rgba(242,140,40,.18); color:#FFD29D; border:1px solid rgba(255,210,157,.35);
            padding:5px 11px; border-radius:99px; font-size:.75rem; font-weight:700; letter-spacing:.08em; margin-bottom:12px; }
.metric-card { background:white; border-radius:15px; padding:17px 18px; min-height:120px;
               box-shadow:0 5px 18px rgba(22,50,79,.06); margin-bottom:12px; }
.metric-label { color:#667085; font-size:.82rem; font-weight:600; }
.metric-value { color:#16324F; font-family:'Manrope',sans-serif; font-size:1.65rem; font-weight:800; margin:5px 0; }
.metric-detail { color:#98A2B3; font-size:.76rem; }
.page-kicker { color:#F28C28; font-weight:800; letter-spacing:.13em; font-size:.72rem; margin-bottom:2px; }
.info-panel { background:#EAF1F7; border-left:4px solid #245B83; padding:13px 16px; border-radius:9px; margin:8px 0 16px; }
.warning-panel { background:#FFF4E8; border-left:4px solid #F28C28; padding:13px 16px; border-radius:9px; margin:8px 0 16px; }
.small-muted { color:#667085; font-size:.86rem; }
div.stButton > button, div.stDownloadButton > button { border-radius:10px; font-weight:700; min-height:2.65rem; }
div.stButton > button[kind="primary"] { background:#F28C28; border-color:#F28C28; color:#142C43; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
hr { border-color:#DCE4ED; }
@media (max-width: 700px) {
 .block-container { padding-left:1rem; padding-right:1rem; }
 .hero { padding:22px 20px; border-radius:16px; }
 .hero h1 { font-size:1.65rem; }
 .metric-value { font-size:1.35rem; }
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏗️ BétonFiabilité")
    st.caption("Laboratoire numérique de fiabilité du béton")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Accueil", "Données expérimentales", "Calcul de résistance", "Analyse statistique",
         "Probabilité de non-atteinte", "Simulation Monte-Carlo", "Comparaison des bétons",
         "Analyse de fiabilité", "Comprendre et apprendre", "Rapports et exportations"],
        label_visibility="visible"
    )
    st.markdown("---")
    if st.button("🧪 Charger l'exemple fictif", use_container_width=True):
        set_demo_data()
        st.success("Exemple fictif chargé.")
    if st.button("↺ Réinitialiser la session", use_container_width=True):
        for key in ["resistances", "fck", "project_name", "lot_name", "age_days",
                    "specimen_type", "data_source", "is_demo", "fck_dashboard",
                    "fck_data", "fck_probability", "project_name_input",
                    "lot_name_input", "age_days_input", "specimen_type_input"]:
            st.session_state.pop(key, None)
        init_state()
        st.rerun()
    st.markdown("---")
    st.caption("Version pédagogique 1.0")
    st.caption("Les calculs ne remplacent pas les essais ni les normes applicables.")

# Header commun
if st.session_state["is_demo"]:
    st.info("🧪 Vous utilisez un jeu de données fictif. Il ne doit pas être présenté comme un résultat de laboratoire réel.")

if page == "Accueil":
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">OUTIL D'ANALYSE • GÉNIE CIVIL</div>
      <h1>Comprendre la variabilité.<br>Mesurer le risque.</h1>
      <p>BétonFiabilité transforme les résultats d'essais de compression en indicateurs statistiques et probabilistes pour étudier le risque qu'une résistance soit inférieure au seuil caractéristique choisi.</p>
    </div>
    """, unsafe_allow_html=True)
    page_header("Tableau de bord", "Vue d'ensemble de votre analyse de résistance à la compression.")
    st.markdown(f"**Analyse :** {st.session_state['project_name']}  ·  **Lot :** {st.session_state['lot_name'] or 'Non renseigné'}  ·  **Source :** {st.session_state['data_source']}")
    st.number_input("Résistance caractéristique exigée fck (MPa)", min_value=0.1,
                    value=float(st.session_state["fck"]), step=0.5, key="fck_dashboard",
                    on_change=sync_widget_to_state, args=("fck_dashboard", "fck"))
    s = current_stats()
    if s:
        p = normal_probability(s["mean"], s["std"], st.session_state["fck"])
        beta = (s["mean"] - st.session_state["fck"]) / s["std"] if s["std"] and s["std"] > 0 else None
        cols = st.columns(4)
        with cols[0]: metric_card("Éprouvettes", f"{s['n']}", "Résultats disponibles")
        with cols[1]: metric_card("Résistance moyenne", f"{s['mean']:.2f} MPa", "Moyenne expérimentale", "#245B83")
        with cols[2]: metric_card("Écart-type", f"{s['std']:.2f} MPa", "Écart-type d'échantillon", "#F28C28")
        with cols[3]: metric_card("P(X < fck)", f"{p:.2%}", "Modèle normal", "#D97706")
        cols2 = st.columns(4)
        with cols2[0]: metric_card("fck", f"{st.session_state['fck']:.2f} MPa", "Seuil étudié")
        with cols2[1]: metric_card("Coefficient de variation", f"{s['cv']:.2f} %" if s["cv"] is not None else "N/D", "σ / μ × 100")
        with cols2[2]: metric_card("Résistance minimale", f"{s['min']:.2f} MPa", "Observée")
        with cols2[3]: metric_card("Indice β", f"{beta:.3f}" if beta is not None else "N/D", "Indicateur normal, non réglementaire")
        st.markdown('<div class="info-panel"><b>Lecture :</b> P(X &lt; fck) est une estimation sous le modèle normal ajusté à la moyenne et à l’écart-type observés. Elle ne constitue pas à elle seule une décision de conformité d’un lot.</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=st.session_state["resistances"], nbinsx=12, name="Résistances observées", marker_color="#4F83A8", opacity=.8))
        fig.add_vline(x=st.session_state["fck"], line_dash="dash", line_color="#F28C28", annotation_text="fck")
        fig.update_layout(title="Distribution des résistances mesurées", xaxis_title="Résistance (MPa)", yaxis_title="Effectif", template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="info-panel"><b>Commencez ici :</b> chargez l’exemple fictif ou ouvrez « Données expérimentales » pour saisir vos propres résultats.</div>', unsafe_allow_html=True)
        st.write("L'application a besoin d'au moins deux résultats individuels pour calculer l'écart-type expérimental.")
        if st.button("Charger l'exemple fictif maintenant", type="primary"):
            set_demo_data()
            st.rerun()
    st.markdown("### Parcours conseillé")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**01 · Données**")
        st.write("Saisissez ou importez les résistances mesurées.")
    with b:
        st.markdown("**02 · Modélisation**")
        st.write("Choisissez une loi et estimez P(X < fck).")
    with c:
        st.markdown("**03 · Interprétation**")
        st.write("Comparez les scénarios et exportez un rapport.")

elif page == "Données expérimentales":
    page_header("Données expérimentales", "Saisir, importer et décrire les résultats de compression.")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Nom du béton / de l'analyse", value=st.session_state["project_name"],
                      key="project_name_input", on_change=sync_widget_to_state,
                      args=("project_name_input", "project_name"))
        st.text_input("Identifiant du lot ou de la gâchée", value=st.session_state["lot_name"],
                      key="lot_name_input", on_change=sync_widget_to_state,
                      args=("lot_name_input", "lot_name"))
        st.number_input("Âge au moment de l'essai (jours)", min_value=1, max_value=3650,
                        value=int(st.session_state["age_days"]), step=1, key="age_days_input",
                        on_change=sync_widget_to_state, args=("age_days_input", "age_days"))
    with col2:
        specimen_options = ["Cylindre", "Cube"]
        st.selectbox("Type d'éprouvette", specimen_options,
                     index=specimen_options.index(st.session_state["specimen_type"]),
                     key="specimen_type_input", on_change=sync_widget_to_state,
                     args=("specimen_type_input", "specimen_type"))
        st.number_input("Résistance caractéristique fck (MPa)", min_value=0.1,
                        value=float(st.session_state["fck"]), step=0.5, key="fck_data",
                        on_change=sync_widget_to_state, args=("fck_data", "fck"))
        st.caption("Ne mélangez pas sans justification des géométries, âges ou protocoles d'essai différents.")
    st.markdown("### Saisie manuelle")
    current_text = "; ".join(f"{v:g}" for v in st.session_state["resistances"])
    raw = st.text_area("Résistances individuelles (MPa)", value=current_text, height=110,
                       placeholder="Exemple : 28.5; 30.2; 31.0; 29.7",
                       help="Séparez les valeurs par un point-virgule ou un retour à la ligne. Utilisez le point pour les décimales.")
    if st.button("Valider les résistances saisies", type="primary"):
        try:
            values = parse_manual_values(raw)
            st.session_state["resistances"] = values.tolist()
            st.session_state["data_source"] = "Saisie manuelle"
            st.session_state["is_demo"] = False
            st.success(f"{len(values)} résistance(s) enregistrée(s) dans la session.")
        except ValueError as exc:
            st.error(str(exc))
    st.markdown("---")
    st.markdown("### Importer un fichier CSV ou Excel")
    uploaded = st.file_uploader("Choisir un fichier", type=["csv", "xlsx", "xls"], key="data_upload")
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                imported_df = pd.read_csv(uploaded, sep=None, engine="python")
            else:
                imported_df = pd.read_excel(uploaded)
            if imported_df.empty:
                st.warning("Le fichier ne contient aucune ligne.")
            else:
                st.dataframe(imported_df.head(12), use_container_width=True)
                numeric_cols = [c for c in imported_df.columns if pd.to_numeric(imported_df[c], errors="coerce").notna().any()]
                if numeric_cols:
                    chosen = st.selectbox("Colonne contenant les résistances (MPa)", numeric_cols)
                    if st.button("Importer cette colonne"):
                        values = pd.to_numeric(imported_df[chosen], errors="coerce").dropna().to_numpy()
                        try:
                            values = validate_resistances(values)
                            st.session_state["resistances"] = values.tolist()
                            st.session_state["data_source"] = f"Import : {uploaded.name} / {chosen}"
                            st.session_state["is_demo"] = False
                            st.success(f"{len(values)} résultat(s) importé(s).")
                        except ValueError as exc:
                            st.error(str(exc))
                else:
                    st.error("Aucune colonne numérique exploitable n'a été détectée.")
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}")
    if st.session_state["resistances"]:
        st.markdown("### Données actuellement utilisées")
        st.dataframe(pd.DataFrame({"N°": range(1, len(st.session_state["resistances"])+1),
                                   "Résistance (MPa)": st.session_state["resistances"]}), use_container_width=True)
        st.caption(f"Source : {st.session_state['data_source']}")

elif page == "Calcul de résistance":
    page_header("Calcul de résistance", "Calculer la résistance à partir de la charge de rupture et de la section chargée.")
    st.markdown('<div class="info-panel">La résistance à la compression est égale à la charge maximale divisée par la section chargée. Avec une charge en kN et une section en mm², le résultat est directement en MPa après conversion de kN en N.</div>', unsafe_allow_html=True)
    geometry = st.radio("Géométrie de l'éprouvette", ["Cylindre", "Cube"], horizontal=True)
    load = st.number_input("Charge de rupture (kN)", min_value=0.001, value=650.0, step=10.0)
    if geometry == "Cylindre":
        dim = st.number_input("Diamètre (mm)", min_value=1.0, value=150.0, step=1.0)
    else:
        dim = st.number_input("Longueur du côté du cube (mm)", min_value=1.0, value=150.0, step=1.0)
    if st.button("Calculer la résistance", type="primary"):
        try:
            result = compressive_strength(load, geometry, dim)
            st.session_state["last_strength_result"] = {
                "strength_mpa": result["strength_mpa"], "load_n": result["load_n"],
                "area_mm2": result["area_mm2"], "area_formula": result["area_formula"],
                "dimensions": result["dimensions"]
            }
        except ValueError as exc:
            st.error(str(exc))
    if "last_strength_result" in st.session_state:
        result = st.session_state["last_strength_result"]
        st.success(f"Dernier calcul : {result['strength_mpa']:.3f} MPa")
        st.latex(r"f_c = \frac{F}{A}")
        st.write(f"Charge convertie : {load:g} kN × 1000 = {result['load_n']:.1f} N")
        st.write(f"Section : {result['area_formula']} → {result['area_mm2']:.2f} mm² ({result['dimensions']})")
        st.write(f"Application numérique : {result['load_n']:.1f} / {result['area_mm2']:.2f} = **{result['strength_mpa']:.3f} MPa**")
        if st.button("Ajouter ce résultat aux données expérimentales", key="add_strength_result"):
            st.session_state["resistances"].append(round(result["strength_mpa"], 6))
            st.session_state["data_source"] = "Données expérimentales + calcul de charge"
            st.session_state["is_demo"] = False
            del st.session_state["last_strength_result"]
            st.success("Résultat ajouté aux données de la session.")
            st.rerun()

elif page == "Analyse statistique":
    page_header("Analyse statistique", "Décrire la dispersion des résistances observées.")
    if len(st.session_state["resistances"]) < 2:
        st.warning("Ajoutez au moins deux résultats dans « Données expérimentales ».")
    else:
        s = current_stats()
        st.markdown("### Indicateurs")
        cols = st.columns(4)
        for col, label, value in [
            (cols[0], "Effectif n", f"{s['n']}"),
            (cols[1], "Moyenne", f"{s['mean']:.3f} MPa"),
            (cols[2], "Écart-type s", f"{s['std']:.3f} MPa"),
            (cols[3], "CV", f"{s['cv']:.2f} %"),
        ]:
            with col: metric_card(label, value)
        table = pd.DataFrame([
            ["Moyenne", s["mean"], "MPa"],
            ["Variance expérimentale", s["variance"], "MPa²"],
            ["Écart-type expérimental", s["std"], "MPa"],
            ["Coefficient de variation", s["cv"], "%"],
            ["Minimum", s["min"], "MPa"],
            ["Maximum", s["max"], "MPa"],
            ["Médiane", s["median"], "MPa"],
            ["Premier quartile Q1", s["q1"], "MPa"],
            ["Troisième quartile Q3", s["q3"], "MPa"],
            ["Résultats strictement inférieurs à fck", s["below_count"], "résultat(s)"],
            ["Proportion empirique sous fck", s["below_fraction"]*100, "%"],
        ], columns=["Indicateur", "Valeur", "Unité"])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("La variance et l'écart-type utilisent la correction d'échantillon (division par n−1). Pour n = 1, ils ne sont pas estimables.")
        x = np.array(st.session_state["resistances"], dtype=float)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=x, nbinsx=min(15, max(5, int(math.sqrt(len(x))*2))),
                                   name="Résultats", marker_color="#4F83A8", opacity=.82))
        fig.add_vline(x=st.session_state["fck"], line_color="#F28C28", line_dash="dash",
                      annotation_text=f"fck = {st.session_state['fck']:.1f} MPa")
        fig.update_layout(title="Histogramme des résistances", xaxis_title="Résistance (MPa)",
                          yaxis_title="Effectif", template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)
        sorted_x = np.sort(x)
        empirical_cdf = np.arange(1, len(x)+1) / len(x)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sorted_x, y=empirical_cdf, mode="lines",
                                  name="Répartition empirique", line_shape="hv",
                                  line=dict(color="#245B83", width=3)))
        fig2.add_vline(x=st.session_state["fck"], line_dash="dash", line_color="#F28C28")
        fig2.update_layout(title="Fonction de répartition empirique", xaxis_title="Résistance (MPa)",
                           yaxis_title="F(x)", yaxis=dict(range=[0,1]), template="plotly_white", height=350)
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Probabilité de non-atteinte":
    page_header("Probabilité de non-atteinte", "Estimer P(X < fck) selon plusieurs hypothèses de distribution.")
    st.number_input("Seuil fck (MPa)", min_value=0.1, value=float(st.session_state["fck"]),
                    step=0.5, key="fck_probability", on_change=sync_widget_to_state,
                    args=("fck_probability", "fck"))
    params = get_probability_parameters()
    if params:
        mean, std, source_label = params
        model = st.selectbox("Modèle probabiliste", ["Normale", "Lognormale", "Empirique (essais uniquement)"])
        if model == "Normale":
            p = normal_probability(mean, std, st.session_state["fck"])
        elif model == "Lognormale":
            p = lognormal_probability(mean, std, st.session_state["fck"])
        else:
            if len(st.session_state["resistances"]) == 0:
                st.error("La méthode empirique nécessite des résistances individuelles.")
                p = None
            else:
                empirical = empirical_probability(st.session_state["resistances"], st.session_state["fck"])
                p = empirical["probability"]
        if p is not None:
            st.markdown("### Résultat")
            metric_card("Probabilité estimée P(X < fck)", f"{p:.4%}", f"Modèle : {model}", "#F28C28")
            if model == "Empirique (essais uniquement)" and len(st.session_state["resistances"]) >= 2:
                empirical_stats = descriptive_statistics(st.session_state["resistances"], st.session_state["fck"])
                shown_mean, shown_std = empirical_stats["mean"], empirical_stats["std"]
                st.write(f"Moyenne expérimentale = **{shown_mean:.3f} MPa** · Écart-type expérimental = **{shown_std:.3f} MPa** · Seuil fck = **{st.session_state['fck']:.3f} MPa**")
            else:
                st.write(f"Moyenne μ = **{mean:.3f} MPa** · Écart-type σ = **{std:.3f} MPa** · Seuil fck = **{st.session_state['fck']:.3f} MPa**")
            if model == "Normale" and std > 0:
                z = (st.session_state["fck"] - mean) / std
                st.latex(r"P(X<f_{ck})=\Phi\left(\frac{f_{ck}-\mu}{\sigma}\right)")
                st.write(f"Valeur normalisée z = ({st.session_state['fck']:.3f} − {mean:.3f}) / {std:.3f} = **{z:.4f}**")
                st.write(f"Indice β = (μ − fck) / σ = **{(mean-st.session_state['fck'])/std:.4f}**")
                negative = float(scipy_stats.norm.cdf(0, loc=mean, scale=std))
                if negative > 0.001:
                    st.warning(f"Le modèle normal attribue {negative:.2%} de probabilité à des résistances négatives. Vérifiez si l'approximation normale est adaptée ; la loi lognormale peut être examinée.")
            elif model == "Lognormale":
                mu_ln, sigma_ln = lognormal_parameters_from_mean_std(mean, std)
                st.write(f"Paramètres logarithmiques : μln = {mu_ln:.5f} et σln = {sigma_ln:.5f}.")
                st.caption("La moyenne et l'écart-type saisis sont les paramètres arithmétiques ; ils sont convertis en paramètres de la loi du logarithme.")
            elif model.startswith("Empirique"):
                st.write(f"{empirical['count']} résultat(s) sur {empirical['n']} sont strictement inférieurs à fck.")
                st.write(f"Intervalle de Wilson à 95 % : [{empirical['ci_low']:.2%} ; {empirical['ci_high']:.2%}].")
            st.markdown("### Visualisation de la distribution")
            if model != "Empirique (essais uniquement)":
                xx, pdf, cdf = distribution_values(mean, std, st.session_state["fck"], model)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xx, y=pdf, mode="lines", name="Densité", line=dict(color="#245B83", width=3)))
                mask = xx < st.session_state["fck"]
                fig.add_trace(go.Scatter(x=xx[mask], y=pdf[mask], mode="lines", fill="tozeroy",
                                         name="Zone X < fck", line=dict(color="#F28C28"), fillcolor="rgba(242,140,40,.25)"))
                fig.add_vline(x=st.session_state["fck"], line_dash="dash", line_color="#F28C28",
                              annotation_text="fck")
                fig.update_layout(title=f"Densité — loi {model.lower()}", xaxis_title="Résistance (MPa)",
                                  yaxis_title="Densité", template="plotly_white", height=390)
                st.plotly_chart(fig, use_container_width=True)
                figc = go.Figure()
                figc.add_trace(go.Scatter(x=xx, y=cdf, mode="lines", name="F(x)", line=dict(color="#245B83", width=3)))
                figc.add_vline(x=st.session_state["fck"], line_dash="dash", line_color="#F28C28")
                figc.add_hline(y=p, line_dash="dot", line_color="#F28C28", annotation_text=f"P = {p:.2%}")
                figc.update_layout(title="Fonction de répartition", xaxis_title="Résistance (MPa)",
                                   yaxis_title="F(x)", yaxis=dict(range=[0,1]), template="plotly_white", height=330)
                st.plotly_chart(figc, use_container_width=True)
            st.markdown('<div class="warning-panel"><b>À retenir :</b> cette probabilité concerne une résistance individuelle modélisée. Elle ne suffit pas à déclarer un lot ou un ouvrage conforme ou non conforme. Les règles d’acceptation dépendent du référentiel applicable et du plan d’échantillonnage.</div>', unsafe_allow_html=True)

elif page == "Simulation Monte-Carlo":
    page_header("Simulation Monte-Carlo", "Reproduire numériquement la variabilité des résistances.")
    st.markdown('<div class="info-panel">Monte-Carlo génère un grand nombre de résistances suivant une loi choisie, puis estime la fréquence des valeurs inférieures à fck. La graine aléatoire permet de reproduire exactement une simulation.</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        mean = st.number_input("Moyenne μ (MPa)", min_value=0.001, value=30.0, step=0.5, key="mc_mu")
        std = st.number_input("Écart-type σ (MPa)", min_value=0.0, value=3.0, step=0.1, key="mc_sigma")
        threshold = st.number_input("Résistance caractéristique fck (MPa)", min_value=0.1, value=float(st.session_state["fck"]), step=0.5, key="mc_fck")
    with cols[1]:
        distribution = st.selectbox("Distribution", ["Normale", "Lognormale"], key="mc_distribution")
        n_sim = st.select_slider("Nombre de simulations", options=[1000, 5000, 10000, 50000, 100000, 250000, 500000], value=100000)
        seed = st.number_input("Graine aléatoire", min_value=0, max_value=2**31-1, value=2026, step=1)
    if st.button("Lancer la simulation", type="primary"):
        try:
            result = simulate_resistances(mean, std, threshold, distribution, n_sim, seed)
            st.session_state["mc_result"] = {
                "probability": result["probability"], "count": result["count"], "n": result["n"],
                "standard_error": result["standard_error"], "ci_low": result["ci_low"],
                "ci_high": result["ci_high"], "mean": mean, "std": std, "fck": threshold,
                "distribution": distribution, "seed": int(seed)
            }
            st.session_state["mc_samples"] = result["samples"][::max(1, len(result["samples"])//20000)].tolist()
        except ValueError as exc:
            st.error(str(exc))
    if "mc_result" in st.session_state:
        r = st.session_state["mc_result"]
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Probabilité simulée", f"{r['probability']:.4%}", "Fréquence X < fck")
        with c2: metric_card("Cas sous le seuil", f"{r['count']:,}".replace(",", " "), f"sur {r['n']:,}".replace(",", " "))
        with c3: metric_card("Erreur-type Monte-Carlo", f"{r['standard_error']:.5f}", "√[p(1−p)/N]")
        st.write(f"Intervalle de Wilson à 95 % : **[{r['ci_low']:.2%} ; {r['ci_high']:.2%}]**")
        st.caption(f"Modèle : {r['distribution']} · μ = {r['mean']} MPa · σ = {r['std']} MPa · fck = {r['fck']} MPa · graine = {r['seed']}")
        samples = np.array(st.session_state["mc_samples"])
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=samples, nbinsx=55, marker_color="#4F83A8", name="Échantillon simulé"))
        fig.add_vline(x=r["fck"], line_dash="dash", line_color="#F28C28", annotation_text="fck")
        fig.update_layout(title="Résistances simulées (échantillon affiché)", xaxis_title="Résistance (MPa)",
                          yaxis_title="Effectif", template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)
        if r["std"] == 0:
            st.info("Cas sans incertitude : toutes les simulations ont la même valeur. La probabilité vaut donc 0 ou 1 selon la position de μ par rapport à fck.")
        if r["distribution"] == "Normale" and r["std"] > 0:
            neg = scipy_stats.norm.cdf(0, loc=r["mean"], scale=r["std"])
            if neg > 0.001:
                st.warning(f"La loi normale génère théoriquement {neg:.2%} de résistances négatives. Ce résultat signale une hypothèse de modèle à examiner.")

elif page == "Comparaison des bétons":
    page_header("Comparaison des bétons", "Comparer plusieurs formulations à seuils et hypothèses explicitement indiqués.")
    st.caption("Les lignes proposées sont des exemples fictifs modifiables, pas des résultats de laboratoire.")
    default_df = pd.DataFrame([
        {"Béton": "Formulation A", "Moyenne μ (MPa)": 30.0, "Écart-type σ (MPa)": 3.0, "Seuil fck (MPa)": 25.0},
        {"Béton": "Formulation B", "Moyenne μ (MPa)": 30.0, "Écart-type σ (MPa)": 5.0, "Seuil fck (MPa)": 25.0},
        {"Béton": "Formulation C", "Moyenne μ (MPa)": 33.0, "Écart-type σ (MPa)": 3.5, "Seuil fck (MPa)": 25.0},
    ])
    if "compare_df" not in st.session_state:
        st.session_state["compare_df"] = default_df
    edited = st.data_editor(st.session_state["compare_df"], num_rows="dynamic", use_container_width=True,
                            column_config={
                                "Moyenne μ (MPa)": st.column_config.NumberColumn(min_value=0.001),
                                "Écart-type σ (MPa)": st.column_config.NumberColumn(min_value=0.0),
                                "Seuil fck (MPa)": st.column_config.NumberColumn(min_value=0.001),
                            })
    st.session_state["compare_df"] = edited
    if st.button("Calculer la comparaison", type="primary"):
        try:
            rows = []
            for _, row in edited.iterrows():
                mu = float(row["Moyenne μ (MPa)"])
                sigma = float(row["Écart-type σ (MPa)"])
                threshold = float(row["Seuil fck (MPa)"])
                p = normal_probability(mu, sigma, threshold)
                beta = (mu-threshold)/sigma if sigma > 0 else None
                rows.append({"Béton": str(row["Béton"]), "μ (MPa)": mu, "σ (MPa)": sigma,
                             "fck (MPa)": threshold, "CV (%)": sigma/mu*100,
                             "P(X < fck) (%)": p*100, "β normal": beta})
            results_df = pd.DataFrame(rows)
            st.session_state["compare_results"] = results_df
        except Exception as exc:
            st.error(f"Vérifiez les valeurs du tableau : {exc}")
    if "compare_results" in st.session_state:
        result_df = st.session_state["compare_results"]
        if result_df.empty:
            st.warning("Ajoutez au moins une formulation dans le tableau.")
        else:
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            fig = px.bar(result_df, x="Béton", y="P(X < fck) (%)", color="Béton",
                         title="Probabilité normale estimée sous le seuil", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Comparaison pédagogique sous hypothèse normale. Une probabilité plus faible n'est pas, à elle seule, une preuve de conformité réglementaire.")

elif page == "Analyse de fiabilité":
    page_header("Analyse de fiabilité", "Étudier le lien entre la marge moyenne, la dispersion et le seuil.")
    params = get_probability_parameters()
    if params:
        mean, std, _ = params
        threshold = st.number_input("Seuil fck (MPa)", min_value=0.1, value=float(st.session_state["fck"]), step=0.5, key="rel_fck")
        if std > 0:
            beta = (mean-threshold)/std
            p = scipy_stats.norm.cdf(-beta)
            c1, c2 = st.columns(2)
            with c1: metric_card("Indice β", f"{beta:.4f}", "(μ − fck) / σ")
            with c2: metric_card("P(X < fck)", f"{p:.4%}", "Sous hypothèse normale", "#F28C28")
            st.latex(r"\beta=\frac{\mu-f_{ck}}{\sigma}")
            st.write("Dans le modèle normal, la probabilité étudiée est Φ(−β). Un β plus élevé correspond à une probabilité plus faible de passer sous le seuil, toutes choses égales par ailleurs.")
        else:
            st.info("Avec σ = 0, il n'y a pas de dispersion dans ce modèle. L'indice β n'est pas défini ; l'événement dépend uniquement de μ par rapport à fck.")
        st.markdown('<div class="warning-panel"><b>Précision :</b> β est ici un indicateur statistique associé au seuil de résistance étudié. Il ne doit pas être présenté comme un indice réglementaire de sécurité structurale sans définir une fonction d’état limite complète, les actions et les résistances correspondantes.</div>', unsafe_allow_html=True)
        st.markdown("### Sensibilité de la probabilité")
        c1, c2 = st.columns(2)
        with c1:
            span_mu = st.slider("Variation de la moyenne autour de μ (%)", 5, 50, 20)
            mu_values = np.linspace(mean*(1-span_mu/100), mean*(1+span_mu/100), 80)
            p_mu = [normal_probability(v, std, threshold) for v in mu_values]
            fig1 = go.Figure(go.Scatter(x=mu_values, y=np.array(p_mu)*100, mode="lines", line=dict(color="#245B83", width=3)))
            fig1.update_layout(title="Influence de la moyenne", xaxis_title="Moyenne μ (MPa)", yaxis_title="P(X < fck) (%)", template="plotly_white", height=330)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            span_s = st.slider("Variation de σ autour de σ (%)", 5, 100, 50)
            s_min = max(0.001, std*(1-span_s/100))
            s_values = np.linspace(s_min, max(std*(1+span_s/100), s_min+0.01), 80)
            p_s = [normal_probability(mean, v, threshold) for v in s_values]
            fig2 = go.Figure(go.Scatter(x=s_values, y=np.array(p_s)*100, mode="lines", line=dict(color="#F28C28", width=3)))
            fig2.update_layout(title="Influence de la dispersion", xaxis_title="Écart-type σ (MPa)", yaxis_title="P(X < fck) (%)", template="plotly_white", height=330)
            st.plotly_chart(fig2, use_container_width=True)

elif page == "Comprendre et apprendre":
    page_header("Comprendre et apprendre", "Les notions, formules et limites nécessaires pour défendre le projet.")
    with st.expander("1. Variable aléatoire et événement étudié", expanded=True):
        st.write("On représente la résistance à la compression par une variable aléatoire X, exprimée en MPa. Pour un seuil fck fixé, l'événement étudié est X < fck.")
        st.latex(r"E=\{X<f_{ck}\}\qquad P(E)=P(X<f_{ck})")
        st.write("Cette probabilité décrit le comportement d'une résistance individuelle tirée du modèle retenu. Elle n'est pas automatiquement la probabilité qu'un lot entier soit rejeté.")
    with st.expander("2. Moyenne, variance et écart-type"):
        st.latex(r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i")
        st.latex(r"s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2\qquad s=\sqrt{s^2}")
        st.write("La moyenne mesure le niveau central des résistances. L'écart-type mesure leur dispersion. La correction n−1 est utilisée pour l'estimation de la variance d'une population à partir d'un échantillon.")
    with st.expander("3. Coefficient de variation"):
        st.latex(r"CV=\frac{s}{\bar{x}}\times100\%")
        st.write("Le coefficient de variation rapporte la dispersion à la moyenne. Il est sans unité et facilite certaines comparaisons de séries.")
    with st.expander("4. Modèle normal"):
        st.latex(r"X\sim\mathcal{N}(\mu,\sigma^2)")
        st.latex(r"P(X<f_{ck})=\Phi\left(\frac{f_{ck}-\mu}{\sigma}\right)")
        st.write("La loi normale est symétrique et peut être une approximation utile. Elle autorise toutefois des valeurs négatives ; cette limite doit être examinée lorsque la dispersion est importante.")
    with st.expander("5. Modèle lognormal"):
        st.write("La loi lognormale modélise une variable strictement positive dont le logarithme suit une loi normale. Si μ et σ sont la moyenne et l'écart-type arithmétiques, les paramètres logarithmiques sont calculés par :")
        st.latex(r"\sigma_{\ln}=\sqrt{\ln\left(1+\frac{\sigma^2}{\mu^2}\right)}")
        st.latex(r"\mu_{\ln}=\ln(\mu)-\frac{1}{2}\sigma_{\ln}^2")
    with st.expander("6. Simulation de Monte-Carlo"):
        st.latex(r"\widehat{P}(X<f_{ck})=\frac{N_{X<f_{ck}}}{N}")
        st.latex(r"SE=\sqrt{\frac{\hat p(1-\hat p)}{N}}")
        st.write("On génère N résistances suivant le modèle choisi, puis on compte celles qui sont inférieures à fck. L'erreur-type diminue en général comme 1/√N. Augmenter N réduit l'erreur numérique de simulation, mais ne corrige pas un mauvais modèle ni des données non représentatives.")
    with st.expander("7. Indice β dans le modèle normal"):
        st.latex(r"\beta=\frac{\mu-f_{ck}}{\sigma}\qquad P(X<f_{ck})=\Phi(-\beta)")
        st.write("Cet indice résume la distance entre la moyenne et le seuil en nombre d'écarts-types. Ici, il ne constitue pas un indice réglementaire de sécurité d'une structure.")
    with st.expander("8. Exemple numérique vérifiable"):
        st.write("Supposons μ = 30 MPa, σ = 3 MPa et fck = 25 MPa. Alors z = (25 − 30)/3 = −1,6667. La probabilité normale est approximativement 4,78 %. Ce calcul peut être vérifié dans la rubrique « Probabilité de non-atteinte ».")
        st.write("Deux bétons de même moyenne peuvent présenter des risques différents : à seuil identique, une dispersion plus grande augmente généralement la probabilité de passer sous le seuil lorsque le seuil est inférieur à la moyenne.")
    st.markdown("---")
    st.markdown("### Résistance caractéristique et fractile de 5 %")
    st.write("Dans de nombreux référentiels de béton, la résistance caractéristique est définie comme un fractile inférieur de 5 % de la distribution de résistance, selon des conditions précises de définition et de contrôle. Cela ne signifie pas que chaque lot peut être déclaré conforme en se basant uniquement sur P(X < fck). Le plan d'échantillonnage et les critères d'acceptation du référentiel applicable restent nécessaires.")

elif page == "Rapports et exportations":
    page_header("Rapports et exportations", "Sauvegarder les analyses et générer des livrables.")
    if len(st.session_state["resistances"]) < 2:
        st.warning("Ajoutez au moins deux résultats avant de générer un rapport statistique.")
    else:
        s = current_stats()
        model = st.selectbox("Modèle utilisé pour le rapport", ["Normale", "Lognormale"])
        if model == "Normale":
            p = normal_probability(s["mean"], s["std"], st.session_state["fck"])
        else:
            p = lognormal_probability(s["mean"], s["std"], st.session_state["fck"])
        st.write(f"Probabilité selon le modèle {model} : **{p:.4%}**")
        source_label = st.session_state["data_source"] + (" — FICTIF" if st.session_state["is_demo"] else "")
        notes = ("Analyse de résistance individuelle sous hypothèse " + model.lower() +
                 ". Données : " + source_label +
                 ". Les résultats dépendent de la représentativité des éprouvettes, de l'âge, de la géométrie et du protocole d'essai.")
        pdf_bytes = build_pdf_report(st.session_state["project_name"], st.session_state["fck"], s, p, model, source_label, notes)
        st.download_button("📄 Télécharger le rapport PDF", data=pdf_bytes,
                           file_name="rapport_betonfiabilite.pdf", mime="application/pdf", type="primary")
        summary = pd.DataFrame([{
            "Projet": st.session_state["project_name"], "Lot": st.session_state["lot_name"],
            "Source": source_label, "fck_MPa": st.session_state["fck"], "n": s["n"],
            "moyenne_MPa": s["mean"], "ecart_type_MPa": s["std"], "CV_pct": s["cv"],
            "modele": model, "probabilite_sous_fck": p
        }])
        csv_buffer = summary.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Télécharger le résumé CSV", data=csv_buffer,
                           file_name="analyse_betonfiabilite.csv", mime="text/csv")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Résumé", index=False)
            pd.DataFrame({"Résistance_MPa": st.session_state["resistances"]}).to_excel(writer, sheet_name="Essais", index=False)
        st.download_button("📊 Télécharger le classeur Excel", data=excel_buffer.getvalue(),
                           file_name="analyse_betonfiabilite.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        record = {
            "id": str(uuid.uuid4()), "saved_at": datetime.now().isoformat(timespec="seconds"),
            "project_name": st.session_state["project_name"], "lot_name": st.session_state["lot_name"],
            "fck": float(st.session_state["fck"]), "age_days": int(st.session_state["age_days"]),
            "specimen_type": st.session_state["specimen_type"],
            "resistances": [float(v) for v in st.session_state["resistances"]],
            "data_source": source_label, "is_demo": bool(st.session_state["is_demo"])
        }
        if st.button("💾 Enregistrer cette analyse sur cet ordinateur"):
            records = load_store()
            records.append(record)
            write_store(records)
            st.session_state["analysis_history"] = records
            st.success(f"Analyse enregistrée dans : {STORE_FILE}")
    st.markdown("---")
    st.markdown("### Sauvegarde JSON")
    export_records = load_store()
    current_record = {
        "project_name": st.session_state["project_name"], "lot_name": st.session_state["lot_name"],
        "fck": float(st.session_state["fck"]), "age_days": int(st.session_state["age_days"]),
        "specimen_type": st.session_state["specimen_type"],
        "resistances": [float(v) for v in st.session_state["resistances"]],
        "data_source": st.session_state["data_source"], "is_demo": bool(st.session_state["is_demo"])
    }
    backup = json.dumps({"format": "betonfiabilite-backup-v1", "current": current_record,
                         "saved_analyses": export_records}, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("Télécharger la sauvegarde JSON", data=backup,
                       file_name="sauvegarde_betonfiabilite.json", mime="application/json")
    restore = st.file_uploader("Restaurer une sauvegarde JSON", type=["json"], key="json_restore")
    if restore is not None and st.button("Restaurer cette sauvegarde"):
        try:
            payload = json.loads(restore.getvalue().decode("utf-8"))
            current = payload.get("current", payload)
            values = validate_resistances(current.get("resistances", []))
            st.session_state["resistances"] = values.tolist()
            st.session_state["fck"] = float(current.get("fck", 25))
            st.session_state["project_name"] = str(current.get("project_name", "Analyse restaurée"))
            st.session_state["lot_name"] = str(current.get("lot_name", ""))
            st.session_state["data_source"] = "Sauvegarde JSON restaurée"
            st.session_state["is_demo"] = bool(current.get("is_demo", False))
            if "saved_analyses" in payload:
                write_store(payload["saved_analyses"])
            st.success("Sauvegarde restaurée dans la session.")
            st.rerun()
        except Exception as exc:
            st.error(f"Sauvegarde invalide : {exc}")
    if export_records:
        st.markdown("### Analyses enregistrées sur cet ordinateur")
        hist_df = pd.DataFrame([{
            "Date": r.get("saved_at", ""), "Projet": r.get("project_name", ""),
            "Lot": r.get("lot_name", ""), "Nombre d'essais": len(r.get("resistances", [])),
            "Fictif": r.get("is_demo", False), "ID": r.get("id", "")
        } for r in export_records])
        st.dataframe(hist_df.drop(columns=["ID"]), use_container_width=True, hide_index=True)
        ids = [r.get("id") for r in export_records]
        chosen_id = st.selectbox("Sélectionner une analyse enregistrée", ids,
                                 format_func=lambda v: next((r.get("project_name", v) for r in export_records if r.get("id")==v), v))
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Charger cette analyse"):
                selected_record = next(r for r in export_records if r.get("id") == chosen_id)
                st.session_state["resistances"] = validate_resistances(selected_record.get("resistances", [])).tolist()
                st.session_state["fck"] = float(selected_record.get("fck", 25.0))
                st.session_state["project_name"] = selected_record.get("project_name", "Analyse chargée")
                st.session_state["lot_name"] = selected_record.get("lot_name", "")
                st.session_state["age_days"] = int(selected_record.get("age_days", 28))
                st.session_state["specimen_type"] = selected_record.get("specimen_type", "Cylindre")
                st.session_state["data_source"] = "Analyse enregistrée rechargée"
                st.session_state["is_demo"] = bool(selected_record.get("is_demo", False))
                sync_widget_defaults()
                st.success("Analyse chargée. Vous pouvez modifier les données puis enregistrer un nouvel état.")
                st.rerun()
        with action_col2:
            if st.button("Supprimer l'analyse sélectionnée"):
                write_store([r for r in export_records if r.get("id") != chosen_id])
                st.success("Analyse supprimée du stockage local.")
                st.rerun()

st.markdown("---")
st.caption("BétonFiabilité · Outil pédagogique d'analyse statistique · Vérifiez toujours le protocole d'essai et le référentiel normatif applicable.")
