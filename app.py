
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Trading Intraday - Simulateur",
    page_icon="📈",
    layout="wide"
)

CAPITAL_INITIAL = 500.0
NB_ACTIONS = 10
COMMISSION = 0.001

# 5 analyses par période de 24 heures
REFRESH_INTERVAL = 17_280_000

ACTIONS = {
    "AIR.PA": "Airbus",
    "OR.PA": "L'Oreal",
    "SAN.PA": "Sanofi",
    "SU.PA": "Schneider Electric",
    "BNP.PA": "BNP Paribas",
    "DG.PA": "Vinci",
    "AI.PA": "Air Liquide",
    "CAP.PA": "Capgemini",
    "VIE.PA": "Veolia",
    "ORA.PA": "Orange",
    "ENGI.PA": "Engie",
    "ACA.PA": "Credit Agricole",
    "RNO.PA": "Renault",
    "STLAM.PA": "Stellantis",
    "GLE.PA": "Societe Generale",
    "HO.PA": "Thales",
    "RI.PA": "Pernod Ricard",
    "MC.PA": "LVMH",
    "KER.PA": "Kering",
    "PUB.PA": "Publicis"
}


# ==========================================================
# INITIALISATION DU PORTEFEUILLE
# ==========================================================

if "cash" not in st.session_state:
    st.session_state.cash = CAPITAL_INITIAL

if "positions" not in st.session_state:
    st.session_state.positions = {}

if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "initialise" not in st.session_state:
    st.session_state.initialise = False

if "market_data" not in st.session_state:
    st.session_state.market_data = {}

if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0


# ==========================================================
# TELECHARGEMENT DES COURS
# ==========================================================

def telecharger_donnees():

    donnees = {}
    erreurs = []

    for ticker in ACTIONS:

        try:
            historique = yf.Ticker(ticker).history(
                period="60d",
                interval="1h",
                auto_adjust=True,
                timeout=15
            )

            if historique is None or historique.empty:
                erreurs.append(ticker)
                continue

            if "Close" not in historique.columns:
                erreurs.append(ticker)
                continue

            close = historique["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = pd.to_numeric(
                close,
                errors="coerce"
            ).dropna()

            if len(close) >= 25:
                donnees[ticker] = close

            else:
                erreurs.append(ticker)

        except Exception as e:
            erreurs.append(ticker)
            print(f"Erreur {ticker}: {e}")

    return donnees, erreurs


# ==========================================================
# ANALYSE DES ACTIONS
# ==========================================================

def analyser_actions(donnees):

    resultats = []

    for ticker, close in donnees.items():

        if len(close) < 25:
            continue

        prix = float(close.iloc[-1])

        if not np.isfinite(prix) or prix <= 0:
            continue

        variation_20h = (
            prix / float(close.iloc[-21]) - 1
        )

        if len(close) >= 121:
            variation_120h = (
                prix / float(close.iloc[-121]) - 1
            )
        else:
            variation_120h = (
                prix / float(close.iloc[0]) - 1
            )

        rendements = close.pct_change().dropna()

        volatilite = float(
            rendements.tail(20).std()
        )

        if not np.isfinite(volatilite):
            volatilite = 0.0

        # Score combinant tendance courte et moyenne
        score = (
            0.60 * variation_20h
            + 0.40 * variation_120h
            - 0.50 * volatilite
        )

        resultats.append({
            "Ticker": ticker,
            "Action": ACTIONS[ticker],
            "Prix (€)": prix,
            "Variation 20h (%)": variation_20h * 100,
            "Variation 120h (%)": variation_120h * 100,
            "Volatilite (%)": volatilite * 100,
            "Score": score
        })

    classement = pd.DataFrame(resultats)

    if classement.empty:
        return classement

    classement = classement.sort_values(
        "Score",
        ascending=False
    ).reset_index(drop=True)

    return classement


# ==========================================================
# FRAIS DE TRANSACTION
# ==========================================================

def calculer_frais(montant):

    return montant * COMMISSION


# ==========================================================
# ACHAT SIMULE
# ==========================================================

def acheter(ticker, prix, places_restantes):

    if places_restantes <= 0:
        return

    cash = float(st.session_state.cash)

    budget = cash / places_restantes

    quantite = budget / (
        prix * (1 + COMMISSION)
    )

    montant = quantite * prix
    frais = calculer_frais(montant)

    total = montant + frais

    if total > cash:
        return

    if quantite <= 0:
        return

    st.session_state.cash -= total

    st.session_state.positions[ticker] = {
        "quantite": quantite,
        "prix_achat": prix
    }

    st.session_state.transactions.append({
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Type": "ACHAT",
        "Action": ACTIONS[ticker],
        "Ticker": ticker,
        "Prix (€)": prix,
        "Quantite": quantite,
        "Montant (€)": montant,
        "Frais (€)": frais
    })


# ==========================================================
# VENTE SIMULEE
# ==========================================================

def vendre(ticker, prix):

    if ticker not in st.session_state.positions:
        return

    position = st.session_state.positions[ticker]

    quantite = position["quantite"]

    montant = quantite * prix

    frais = calculer_frais(montant)

    produit_vente = montant - frais

    st.session_state.cash += produit_vente

    st.session_state.transactions.append({
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Type": "VENTE",
        "Action": ACTIONS[ticker],
        "Ticker": ticker,
        "Prix (€)": prix,
        "Quantite": quantite,
        "Montant (€)": montant,
        "Frais (€)": frais
    })

    del st.session_state.positions[ticker]


# ==========================================================
# GESTION AUTOMATIQUE DU PORTEFEUILLE
# ==========================================================

def gerer_portefeuille(classement, donnees):

    if classement.empty:
        return

    selection = classement.head(NB_ACTIONS)["Ticker"].tolist()

    prix_actuels = {
        ticker: float(series.iloc[-1])
        for ticker, series in donnees.items()
        if len(series) > 0
    }

    # Premier investissement
    if not st.session_state.initialise:

        nombre = min(
            NB_ACTIONS,
            len(selection)
        )

        for ticker in selection[:nombre]:

            prix = prix_actuels.get(ticker)

            if prix is not None:
                acheter(
                    ticker,
                    prix,
                    nombre - len(st.session_state.positions)
                )

        if st.session_state.positions:
            st.session_state.initialise = True

        return

    # Vente des actions qui ne sont plus selectionnees
    anciennes_positions = list(
        st.session_state.positions.keys()
    )

    for ticker in anciennes_positions:

        if ticker not in selection:

            prix = prix_actuels.get(ticker)

            if prix is not None:
                vendre(ticker, prix)

    # Achat des nouvelles actions selectionnees
    for ticker in selection:

        if ticker in st.session_state.positions:
            continue

        if len(st.session_state.positions) >= NB_ACTIONS:
            break

        prix = prix_actuels.get(ticker)

        if prix is not None:

            places = (
                NB_ACTIONS
                - len(st.session_state.positions)
            )

            acheter(ticker, prix, places)


# ==========================================================
# INTERFACE
# ==========================================================

st.title("📈 Simulateur de Trading Intraday")

st.caption(
    "Portefeuille virtuel de 500 € - "
    "Selection automatique et arbitrages simules"
)

if AUTO_REFRESH_AVAILABLE:

    compteur = st_autorefresh(
        interval=REFRESH_INTERVAL,
        key="actualisation_automatique"
    )

    st.caption(
        "Actualisation automatique : environ "
        "5 fois par 24 heures."
    )

else:

    st.warning(
        "Actualisation automatique indisponible. "
        "Utilisez le bouton d'actualisation."
    )


# ==========================================================
# COMMANDES
# ==========================================================

col_a, col_b = st.columns(2)

with col_a:
    actualiser = st.button(
        "🔄 Analyser maintenant",
        use_container_width=True
    )

with col_b:
    reinitialiser = st.button(
        "♻️ Reinitialiser les 500 €",
        use_container_width=True
    )

if reinitialiser:

    st.session_state.cash = CAPITAL_INITIAL
    st.session_state.positions = {}
    st.session_state.transactions = []
    st.session_state.initialise = False
    st.session_state.market_data = {}
    st.session_state.last_fetch = None
    st.session_state.scan_count = 0

    st.rerun()


# ==========================================================
# TELECHARGEMENT ET ANALYSE
# ==========================================================

maintenant = datetime.now()

dernier = st.session_state.last_fetch

telecharger = (
    dernier is None
    or actualiser
    or (maintenant - dernier) >= timedelta(
        milliseconds=REFRESH_INTERVAL
    )
)

if telecharger:

    with st.spinner(
        "Telechargement des cours et analyse des actions..."
    ):

        donnees, erreurs = telecharger_donnees()

        if donnees:

            classement = analyser_actions(donnees)

            st.session_state.market_data = donnees
            st.session_state.classement = classement
            st.session_state.erreurs = erreurs
            st.session_state.last_fetch = datetime.now()
            st.session_state.scan_count += 1

            gerer_portefeuille(
                classement,
                donnees
            )

        else:

            st.error(
                "Impossible de recuperer les cours. "
                "Yahoo Finance ne renvoie aucune donnee exploitable."
            )

else:

    donnees = st.session_state.market_data
    classement = st.session_state.get(
        "classement",
        pd.DataFrame()
    )
    erreurs = st.session_state.get("erreurs", [])


# ==========================================================
# AFFICHAGE DES RESULTATS
# ==========================================================

if not st.session_state.market_data:

    st.info(
        "En attente des premieres donnees boursieres."
    )

    st.stop()


donnees = st.session_state.market_data

classement = st.session_state.get(
    "classement",
    pd.DataFrame()
)

prix_actuels = {
    ticker: float(series.iloc[-1])
    for ticker, series in donnees.items()
    if len(series) > 0
}


# Valeur du portefeuille
valeur_positions = 0.0

lignes_positions = []

for ticker, position in st.session_state.positions.items():

    prix = prix_actuels.get(
        ticker,
        position["prix_achat"]
    )

    quantite = position["quantite"]

    valeur = quantite * prix

    cout = quantite * position["prix_achat"]

    plus_value = valeur - cout

    valeur_positions += valeur

    lignes_positions.append({
        "Action": ACTIONS[ticker],
        "Ticker": ticker,
        "Quantite": round(quantite, 5),
        "Prix achat (€)": round(position["prix_achat"], 2),
        "Cours actuel (€)": round(prix, 2),
        "Valeur (€)": round(valeur, 2),
        "Plus-value (€)": round(plus_value, 2)
    })


cash = float(st.session_state.cash)

valeur_totale = cash + valeur_positions

benefice = valeur_totale - CAPITAL_INITIAL

performance = (
    benefice / CAPITAL_INITIAL
) * 100


# ==========================================================
# INDICATEURS PRINCIPAUX
# ==========================================================

st.subheader("Mon portefeuille")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Capital initial",
    f"{CAPITAL_INITIAL:.2f} €"
)

c2.metric(
    "Valeur actuelle",
    f"{valeur_totale:.2f} €"
)

c3.metric(
    "Benefice / Perte",
    f"{benefice:+.2f} €",
    f"{performance:+.2f} %"
)

c4.metric(
    "Liquidites disponibles",
    f"{cash:.2f} €"
)

st.divider()


# ==========================================================
# POSITIONS
# ==========================================================

st.subheader("Actions detenues")

if lignes_positions:

    tableau_positions = pd.DataFrame(
        lignes_positions
    )

    st.dataframe(
        tableau_positions,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Aucune action achetee pour le moment."
    )


# ==========================================================
# CLASSEMENT DES ACTIONS
# ==========================================================

st.divider()

st.subheader("Classement automatique des actions")

st.write(
    "Les actions sont classees selon leur tendance "
    "sur 20 et 120 heures et leur volatilite."
)

if not classement.empty:

    affichage = classement.copy()

    affichage["Prix (€)"] = affichage["Prix (€)"].round(2)

    for colonne in [
        "Variation 20h (%)",
        "Variation 120h (%)",
        "Volatilite (%)"
    ]:
        affichage[colonne] = affichage[colonne].round(2)

    affichage["Score"] = affichage["Score"].round(4)

    st.dataframe(
        affichage,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Selection pour le prochain portefeuille")

    selection = classement.head(NB_ACTIONS)

    for _, ligne in selection.iterrows():

        st.write(
            f"**{ligne['Action']}** "
            f"({ligne['Ticker']}) - "
            f"Score : {ligne['Score']:.4f}"
        )


# ==========================================================
# HISTORIQUE DES TRANSACTIONS
# ==========================================================

st.divider()

st.subheader("Historique des achats et ventes")

if st.session_state.transactions:

    historique = pd.DataFrame(
        st.session_state.transactions
    )

    st.dataframe(
        historique.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    total_frais = historique["Frais (€)"].sum()

    st.metric(
        "Total des commissions payees",
        f"{total_frais:.2f} €"
    )

else:

    st.info(
        "Les transactions apparaitront ici."
    )


# ==========================================================
# INFORMATIONS TECHNIQUES
# ==========================================================

st.divider()

st.caption(
    f"Analyses effectuees : {st.session_state.scan_count}"
)

if st.session_state.last_fetch:

    st.caption(
        "Derniere analyse : "
        + st.session_state.last_fetch.strftime(
            "%d/%m/%Y a %H:%M:%S"
        )
    )

if erreurs:

    st.warning(
        "Certaines actions n'ont pas fourni de donnees : "
        + ", ".join(erreurs)
    )

st.caption(
    "Simulation uniquement. Les performances passees "
    "ne garantissent pas les performances futures. "
    "Les cours proviennent de Yahoo Finance."
)
