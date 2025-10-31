# Mettre en pratique l'IA générative en finance

Ce dossier Repository est lié au cours `Mettre en pratique l'IA générative en finance`. Le cours entier est disponible sur [LinkedIn Learning][lil-course-url].

![Nom final de la formation][lil-thumbnail-url] 

Destinée aux analystes financiers, data analysts, data scientists, professionnels de la conformité ou toute personne souhaitant appliquer l’IA générative à la finance, cette formation vous permettra d’acquérir les bases pratiques dans un contexte métier. Vous apprendrez à préparer votre environnement technique, cadrer un projet d’IA générative, rédiger et tester des prompts efficaces pour l’analyse financière, et construire un chatbot financier basé sur le RAG pour interroger des rapports complexes (texte, tableaux, PDF). Accompagné par Natacha Njongwa Yepnga, vous verrez également comment transformer ces approches en mini-applications concrètes qui font gagner du temps en finance, comptabilité et conformité. Vous disposerez ainsi d’un socle opérationnel solide, directement appliqué à vos futurs projets IA en entreprise.

Ce cours est intégré à GitHub Codespaces, un environnement de développement instantané « dans le nuage » qui offre toutes les fonctionnalités de votre IDE préféré sans nécessiter de configuration sur une machine locale. Avec Codespaces, vous pouvez vous exercer à partir de n'importe quelle machine, à tout moment, tout en utilisant un outil que vous êtes susceptible de rencontrer sur votre lieu de travail. Consultez la vidéo "Utiliser Codespaces sur GitHub" pour savoir comment démarrer

Vous trouverez ici des notebooks et fichiers d’exercice permettant de mettre en pratique :
- Le prompt engineering appliqué à la finance
- Le RAG (Retrieval-Augmented Generation) avec données financières
- Le développement d’applications interactives avec Gradio et Streamlit

Cette formation vous permettra de comprendre comment l’intelligence artificielle générative transforme le secteur financier et comment concevoir vos propres outils d’analyse basés sur des modèles de langage (LLMs).


## Instructions

Ce dossier repository contient l’ensemble des fichiers utilisés dans la formation.
Il ne comporte aucune branche spécifique : tout le contenu se trouve dans la branche principale (main).

1. Téléchargez le dépôt via le bouton "Code > Download ZIP",
   ou clonez-le avec la commande :
      git clone https://github.com/LinkedInLearning/GenAI-en-finance_5491062.git

2. Installez les dépendances nécessaires :
      pip install -r requirements.txt

3. Lancez les applications :
   - Application Gradio :
        python app.py
     L’application sera disponible sur : http://127.0.0.1:7860

   - Application Streamlit :
        streamlit run main.py
     L’application s’ouvrira automatiquement dans votre navigateur.


## Installation

1. Pour utiliser ces fichiers d’exercice, vous avez besoin de :
   - Python 3.10 ou version supérieure
   - Des bibliothèques listées dans requirements.txt
     (exemples : openai, gradio, streamlit, pandas, numpy, yfinance, etc.)

2. Clonez ce dossier repository sur votre machine locale (Mac, CMD Windows ou outil GUI tel que SourceTree).

3. Suivez les instructions spécifiques dans les fichiers de code pour exécuter les exemples.


### Formateur
 
**Natacha NJONGWA YEPNGA**  
Data Scientist • Quantitative Analyst • Fondatrice de LDA Advisory

Retrouvez mes autres formations sur [LinkedIn Learning][lil-URL-trainer].

[0]: # (Replace these placeholder URLs with actual course URLs)
[lil-course-url]: https://www.linkedin.com
[lil-thumbnail-url]: https://media.licdn.com/dms/image/v2/D4E0DAQG0eDHsyOSqTA/learning-public-crop_675_1200/B4EZVdqqdwHUAY-/0/1741033220778?e=2147483647&v=beta&t=FxUDo6FA8W8CiFROwqfZKL_mzQhYx9loYLfjN-LNjgA
[lil-URL-trainer]: https://www.linkedin.com/learning/instructors/natacha-njongwa-yepnga

[1]: # (End of FR-Instruction ###############################################################################################)
