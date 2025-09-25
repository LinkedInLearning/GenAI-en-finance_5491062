# -*- coding: utf-8 -*-
"""
Streamlit – Analyse IA des Marchés : KPIs Yahoo Finance & Recommandations
Robuste : tests intégrés, compatibilité OpenAI 1.x et 0.28, fallback modèles.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Analyse IA des Marchés", page_icon="📊", layout="wide")

# -----------------------------
# ENV & versions
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")

def get_versions() -> Tuple[str, str]:
    """Retourne versions de streamlit et openai si possible."""
    try:
        import openai as openai_legacy  # ancienne API
        o_ver = getattr(openai_legacy, "__version__", "unknown")
    except Exception:
        o_ver = "non importé"
    return st.__version__, o_ver

st_ver, openai_ver = get_versions()

# -----------------------------
# Prompt (analyse en paragraphes)
# -----------------------------
PROMPT_TEMPLATE = """Tu es analyste financier senior.

CONTEXTE
{context}

DONNÉES (Yahoo Finance)
{data_desc}

OBJECTIF
Rédige une analyse financière claire, fluide et actionnable basée uniquement sur les données fournies.
Ton destinataire est un décideur non spécialiste : il attend un texte compréhensible, avec des phrases courtes mais informatives.

CONSIGNES D’ÉCRITURE
- Langue : français. Style professionnel, fluide et direct.
- Écris en paragraphes, pas en puces ni en JSON.
- N’invente rien : si une donnée est manquante, mentionne "inconnu".
- Cite toujours les chiffres avec unité/devise et période (ex. 185,42 USD, 3,20 %, 1M, 3M, volatilité annualisée 20j).
- Mets en avant les faits chiffrés pour appuyer chaque analyse.
- Structure ton texte en sections avec titres (Résumé, KPIs clés, Risques, Opportunités, Recommandations).
- Chaque section doit contenir 3–6 phrases rédigées.

CONTRAINTE FINALE
- Réponds uniquement en TEXTE rédigé, structuré en paragraphes avec titres.
- Pas de JSON, pas de code, pas de listes à puces.
"""

# -----------------------------
# OpenAI client – support 1.x et 0.28
# -----------------------------
def make_chat_request(prompt: str, preferred_models=None) -> str:
    """
    Envoie 'prompt' à l'API OpenAI.
    - Essaie d'abord la nouvelle lib (openai>=1.x) via OpenAI().chat.completions.create
    - Sinon bascule sur l'ancienne (openai==0.28) via openai.ChatCompletion.create
    - Fallback de modèle: gpt-5 -> gpt-4o -> gpt-4o-mini
    """
    if preferred_models is None:
        preferred_models = ["gpt-5", "gpt-4o", "gpt-4o-mini"]

    # 1) Tentative nouvelle API
    try:
        from openai import OpenAI  # nouvelle API
        if not API_KEY:
            return "[ERREUR] OPENAI_API_KEY introuvable (fichier .env manquant ?)."
        client = OpenAI(api_key=API_KEY)
        last_exc = None
        for mdl in preferred_models:
            try:
                resp = client.chat.completions.create(
                    model=mdl,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                continue
        return f"[ERREUR API OpenAI] {type(last_exc).__name__}: {last_exc}"
    except Exception:
        # 2) Fallback ancienne API
        try:
            import openai  # ancienne API
            if not API_KEY:
                return "[ERREUR] OPENAI_API_KEY introuvable (fichier .env manquant ?)."
            openai.api_key = API_KEY
            last_exc = None
            for mdl in preferred_models:
                try:
                    resp = openai.ChatCompletion.create(
                        model=mdl,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    last_exc = e
                    continue
            return f"[ERREUR API OpenAI] {type(last_exc).__name__}: {last_exc}"
        except Exception as e2:
            return f"[ERREUR] Impossible d'importer la librairie OpenAI: {e2}"

# -----------------------------
# Data utils
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_adj_close(universe: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    data = yf.download(universe, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        prices = data[["Adj Close"]].copy()
        if len(universe) == 1:
            prices.columns = [universe[0]]
    return prices

def compute_kpis(prices: pd.DataFrame, win: int = 20) -> pd.DataFrame:
    prices = prices.dropna(how="all")
    rets = np.log(prices).diff()

    rows = []
    for col in prices.columns:
        s = prices[col].dropna()
        r = rets[col].dropna()
        last_price = float(s.iloc[-1]) if len(s) else np.nan
        r_1m = float(np.exp(r.iloc[-21:].sum()) - 1) if len(r) >= 21 else np.nan
        r_3m = float(np.exp(r.iloc[-63:].sum()) - 1) if len(r) >= 63 else np.nan
        vol20 = float(r.iloc[-win:].std() * np.sqrt(252)) if len(r) >= win else np.nan
        rows.append(dict(
            ticker=col,
            last_price=last_price,
            return_1m=r_1m,
            return_3m=r_3m,
            vol20_annualized=vol20
        ))
    return pd.DataFrame(rows)

def kpis_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "ticker": "", "last_price": "NA",
            "return_1m": "NA", "return_3m": "NA", "vol20_annualized": "NA"
        }])
    k = df.copy()
    k["last_price"]       = k["last_price"].map(lambda x: f"{x:,.2f} USD" if pd.notnull(x) else "NA")
    k["return_1m"]        = k["return_1m"].map(lambda x: f"{x*100:.2f} %" if pd.notnull(x) else "NA")
    k["return_3m"]        = k["return_3m"].map(lambda x: f"{x*100:.2f} %" if pd.notnull(x) else "NA")
    k["vol20_annualized"] = k["vol20_annualized"].map(lambda x: f"{x*100:.2f} %" if pd.notnull(x) else "NA")
    return k[["ticker", "last_price", "return_1m", "return_3m", "vol20_annualized"]]

def kpis_for_prompt(df: pd.DataFrame) -> str:
    if df.empty:
        return "KPIs (Yahoo Finance):\nAucune donnée disponible."
    return "KPIs (Yahoo Finance):\n" + kpis_display(df).to_string(index=False)

# -----------------------------
# UI – Sidebar
# -----------------------------
st.title("Analyse IA des Marchés : KPIs Yahoo Finance & Recommandations")
st.caption("De la donnée brute à une note d’analyse exploitable pour la décision.")

with st.sidebar:
    st.header("Paramètres")
    tickers_csv = st.text_input("Tickers (CSV)", value="AAPL,MSFT,JPM,XOM")
    c1, c2 = st.columns(2)
    with c1:
        start = st.text_input("Début (YYYY-MM-DD)", value="2020-01-01")
    with c2:
        end = st.text_input("Fin (YYYY-MM-DD ou vide)", value="")

    context = st.text_area("Contexte", value="Portefeuille multi-secteurs, horizon court terme.", height=80)
    extra   = st.text_area("Consignes additionnelles", value="Souligne les divergences sectorielles et propose 3 actions.", height=80)

    st.markdown("---")
    st.subheader("Diagnostics rapides")
    st.text(f"Streamlit {st_ver} | OpenAI {openai_ver}")
    st.text(f"API key détectée: {'oui' if API_KEY else 'non'}")
    if API_KEY:
        st.text(f"Préfixe clé: {API_KEY[:4]}****")
    test_api = st.button("Tester OpenAI")
    test_yf  = st.button("Tester Yahoo Finance")
    st.markdown("---")
    run = st.button("Lancer l’analyse")

# -----------------------------
# Diagnostics
# -----------------------------
if test_api:
    st.info("Test OpenAI en cours…")
    sample = "Réponds en un mot : OK."
    out = make_chat_request(sample, preferred_models=["gpt-5", "gpt-4o", "gpt-4o-mini"])
    st.write(out)

if test_yf:
    st.info("Test Yahoo Finance en cours…")
    try:
        df = fetch_adj_close(["AAPL"], "2023-01-01", None)
        st.write(df.tail())
    except Exception as e:
        st.error(f"Échec yfinance: {e}")

# -----------------------------
# Corps – Pipeline
# -----------------------------
if run:
    # Parse tickers
    universe = [t.strip().upper() for t in (tickers_csv or "").split(",") if t.strip()]
    if not universe:
        st.error("Fournis au moins un ticker (ex. AAPL,MSFT).")
        st.stop()

    start = start or "2020-01-01"
    end = end if end not in ("", None) else None

    # 1) Yahoo Finance
    with st.spinner("Téléchargement des prix ajustés…"):
        try:
            prices = fetch_adj_close(universe, start, end)
        except Exception as e:
            st.error(f"[ERREUR yfinance] {type(e).__name__}: {e}")
            st.stop()

    if prices.dropna(how="all").empty:
        st.info("Aucune donnée disponible pour ces paramètres.")
        st.stop()

    # 2) KPIs
    df_kpis = compute_kpis(prices)
    df_display = kpis_display(df_kpis)
    data_desc = kpis_for_prompt(df_kpis)

    tab1, tab2, tab3 = st.tabs(["Analyse IA", "KPIs", "Graphique"])

    with tab2:
        st.subheader("KPIs (formatés)")
        st.dataframe(df_display, use_container_width=True)

    with tab3:
        st.subheader("Prix ajustés (Adj Close)")
        fig, ax = plt.subplots()
        prices.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix")
        ax.set_title("Adj Close")
        st.pyplot(fig, clear_figure=True)

    # 3) Appel modèle
    with st.spinner("Génération de l’analyse (gpt-5 → fallback si indisponible)…"):
        prompt = PROMPT_TEMPLATE.format(context=context, data_desc=data_desc)
        if extra:
            prompt += "\n\nConsignes additionnelles :\n" + extra
        analysis = make_chat_request(prompt, preferred_models=["gpt-5", "gpt-4o", "gpt-4o-mini"])

    with tab1:
        st.subheader("Analyse rédigée")
        st.write(analysis)

else:
    st.info("Renseigne les paramètres dans la barre latérale, puis clique **Lancer l’analyse**.")
