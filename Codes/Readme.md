# IA Générative appliquée à la Finance

Ce dépôt accompagne la **formation "IA Générative appliquée à la finance"**.  
Vous y trouverez des notebooks, ainsi que deux types d’applications (Gradio et Streamlit) permettant de mettre en pratique :

- Le **Prompt Engineering** appliqué à la finance
- Le **RAG (Retrieval-Augmented Generation)** avec données financières

---

## Structure du projet

```
Codes/
│
├── 00_PromptEng/                # Partie 1 : Prompt Engineering
│   ├── 01_ia_finance_prompt_app.ipynb   # Notebook 
│   ├── app.py                   # Application Gradio
│   ├── main.py                  # Application Streamlit
│   └── requirements.txt         # Dépendances spécifiques
│
├── 01_RAG/                      # Partie 2 : Retrieval-Augmented Generation
│   ├── 02_rag_finance_prompt_first_then_rag.ipynb
│   ├── app.py                   # Application Gradio
│   ├── main.py                  # Application Streamlit
│   └── .env                     # Variables d’environnement (API keys, etc.)
│
├── data/                        # Jeux de données financiers 
│
└── Readme.md                    # Documentation principale
```


##  Lancer les applications

### 1. Application **Gradio** (`app.py`)

Interface simple et rapide pour tester un modèle en finance :  
```bash
python app.py
```
➡️ L’application sera disponible sur : `http://127.0.0.1:7860`

---

### 2. Application **Streamlit** (`main.py`)

Interface plus complète et interactive pour explorer les cas d’usage :  
```bash
streamlit run main.py
```
➡️ L’application s’ouvrira automatiquement dans votre navigateur.



##  Installation de l’environnement virtuel

### Option 1 – Conda
```bash
conda create -n ia_finance python=3.10 -y
conda activate ia_finance
pip install -r requirements.txt
```

### Option 2 – venv
```bash
python -m venv ia_finance
.\ia_finance\Scripts\activate     # Windows
source ia_finance/bin/activate    # Mac/Linux
pip install -r requirements.txt
```



## Dépendances principales

- `openai` – API LLMs  
- `gradio` – Création d’interfaces rapides  
- `streamlit` – Développement d’applications data interactives  
- `pypdf`, `PyMuPDF` – Lecture de documents financiers PDF  
- `ollama` – Modèles locaux pour le RAG  
- `pandas`, `numpy`, `matplotlib`, `yfinance` – Analyse et visualisation de données financières  

Toutes les dépendances sont listées dans requirements.txt.




##  Auteur

**Natacha NJONGWA YEPNGA**  
Data Scientist | Quantitative Analyst | Fondatrice de LDA Advisory  

 [YouTube – LeCoinStat](https://www.youtube.com/@lecoinstat)  


---
