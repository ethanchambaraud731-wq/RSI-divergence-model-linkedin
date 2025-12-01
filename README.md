# RSI Divergence Model - Tableau de Bord Streamlit

## 📝 Description

Ce projet est une application Streamlit qui permet d’analyser la dynamique d’un actif financier à l’aide de l’indicateur **RSI (Relative Strength Index)**.
L’objectif est de visualiser les points de surachat/survente, détecter des **divergences haussières et baissières**, générer des signaux d’achat et de vente, et simuler l’évolution du capital de l’utilisateur en fonction des trades.

---

## ⚡ Fonctionnalités

* Téléchargement automatique des données via **Yahoo Finance** (`yfinance`)
* Calcul du **RSI** sur différentes périodes paramétrables
* Détection des **divergences prix / RSI**
* Signaux automatiques **Buy / Sell**
* Simulation du **capital et PnL total**
* Visualisation interactive :

  * Graphique prix + RSI + signaux
  * Courbe de l’évolution du capital
  * Tableaux détaillés des trades et signaux

---

## 🛠️ Installation

1. **Cloner le repository :**

```bash
git clone https://github.com/<votre-utilisateur>/<votre-repo>.git
cd <votre-repo>
```

2. **Installer les dépendances :**

```bash
pip install -r requirements.txt
```

3. **Lancer l’application :**

```bash
streamlit run streamlit_app.py
```

---

## ⚙️ Utilisation

* Configurez les paramètres dans la barre latérale :

  * Ticker (ex. NVDA)
  * Période RSI
  * Plage de données
  * Détection de divergence
  * Capital initial

* Cliquez sur **Calculer** pour générer :

  * Graphiques de prix et RSI
  * Signal Buy/Sell
  * Tableau de PnL et capital

---

## ⚖️ Limites du modèle

* RSI = indicateur rétrospectif : faible réactivité sur marchés très volatils
* Seuils 30/70 statiques : peuvent générer de faux signaux
* Efficacité dépend fortement du paramétrage et des données

> À utiliser comme **outil d’analyse complémentaire**, pas comme système de trading autonome.

---

## 📂 Structure du projet

```
├─ streamlit_app.py       # Script principal Streamlit
├─ requirements.txt       # Dépendances Python
├─ README.md              # Documentation
└─ (optionnel) modules/   # Modules additionnels ou fichiers de configuration
```

---

## 🔗 Liens utiles

* [Streamlit Documentation](https://docs.streamlit.io)
* [yfinance Documentation](https://pypi.org/project/yfinance/)
* [Plotly Documentation](https://plotly.com/python/)
