# -*- coding: utf-8 -*-
"""
Analyse IA des Marchés : KPIs Yahoo Finance & Recommandations
Compatibilité Gradio 3.x : pas de unsafe_allow_html, pas de Pyplot.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gradio as gr
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# Config
# =========================
plt.rcParams["figure.figsize"] = (9, 4.5)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["grid.color"] = "#E6E6E6"

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
client: Optional[OpenAI] = OpenAI(api_key=API_KEY) if API_KEY else None

# =========================
# Prompt
# =========================
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
"""

# =========================
# Data utils
# =========================
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
            "ticker": "", "last_price": "NA", "return_1m": "NA", "return_3m": "NA", "vol20_annualized": "NA"
        }])
    k = df.copy()
    k["last_price"]       = k["last_price"].map(lambda x: f"{x:,.2f} USD" if pd.notnull(x) else "NA")
    k["return_1m"]        = k["return_1m"].map(lambda x: f"{x*100:.2f} %" if pd.notnull(x) else "NA")
    k["return_3m"]        = k["return_3m"].map(lambda x: f"{x*100:.2f} %" if pd.notnull(x) else "NA")
    k["vol20_annualized"] = k["vol20_annualized"].map(lambda x: f"{x*100:.2f} %" if pd.notnull(x) else "NA")
    return k

def kpis_for_prompt(df: pd.DataFrame) -> str:
    if df.empty:
        return "KPIs (Yahoo Finance):\nAucune donnée disponible."
    return "KPIs (Yahoo Finance):\n" + kpis_display(df).to_string(index=False)

def plot_prices(prices: pd.DataFrame):
    fig, ax = plt.subplots()
    prices.plot(ax=ax)
    ax.set_title("Prix ajustés (Adj Close)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix")
    fig.tight_layout()
    return fig

# =========================
# OpenAI
# =========================
def run_prompt(context: str, data_desc: str, extra: str = "", model="gpt-5") -> str:
    if client is None:
        return "[ERREUR] OPENAI_API_KEY introuvable."
    prompt = PROMPT_TEMPLATE.format(context=context, data_desc=data_desc)
    if extra:
        prompt += "\n\nConsignes additionnelles :\n" + extra
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERREUR API OpenAI] {e}"

# =========================
# Pipeline
# =========================
def ui_pipeline(tickers_csv, start, end, context, extra):
    universe = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
    if not universe:
        return "[ERREUR] Fournis au moins un ticker.", pd.DataFrame(), "", None

    start = start or "2020-01-01"
    end = end if end not in ("", None) else None

    try:
        prices = fetch_adj_close(universe, start, end)
    except Exception as e:
        return f"[ERREUR yfinance] {e}", pd.DataFrame(), "", None

    if prices.dropna(how="all").empty:
        return "[INFO] Aucune donnée disponible.", pd.DataFrame(), "", None

    df_kpis = compute_kpis(prices)
    kpis_table = kpis_display(df_kpis)
    data_desc = kpis_for_prompt(df_kpis)
    analysis = run_prompt(context, data_desc, extra=extra, model="gpt-5")
    fig = plot_prices(prices)

    return analysis, kpis_table, data_desc, fig

# =========================
# UI Gradio
# =========================
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Analyse IA des Marchés : KPIs Yahoo Finance & Recommandations")
        gr.Markdown("**De la donnée brute à une note d’analyse exploitable pour la décision.**")

        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                gr.Markdown("### Paramètres")
                tickers_in = gr.Textbox(label="Tickers (CSV)", value="AAPL,MSFT,JPM,XOM")
                with gr.Row():
                    start_in = gr.Textbox(label="Début (YYYY-MM-DD)", value="2020-01-01")
                    end_in   = gr.Textbox(label="Fin (YYYY-MM-DD ou vide)", value="")
                context_in = gr.Textbox(label="Contexte", value="Portefeuille multi-secteurs, horizon court terme.", lines=2)
                extra_in   = gr.Textbox(label="Consignes additionnelles", value="Souligne les divergences sectorielles et propose 3 actions.", lines=2)
                run_btn = gr.Button("Lancer l’analyse")

                gr.Markdown("**Modèle utilisé :** gpt-5")

            with gr.Column(scale=2):
                gr.Markdown("### Résultats")
                with gr.Tabs():
                    with gr.TabItem("Analyse rédigée"):
                        out_text = gr.Textbox(label="Analyse (IA)", lines=20)
                    with gr.TabItem("KPIs (tableau)"):
                        out_table = gr.Dataframe(label="KPIs formatés")
                    with gr.TabItem("Graphique"):
                        out_plot = gr.Plot(label="Graphique")
                    with gr.TabItem("Bloc KPIs injecté"):
                        out_prompt = gr.Code(label="Bloc KPIs pour le prompt", language="markdown")

        run_btn.click(
            fn=ui_pipeline,
            inputs=[tickers_in, start_in, end_in, context_in, extra_in],
            outputs=[out_text, out_table, out_prompt, out_plot],
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
