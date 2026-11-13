
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# SIMULATEUR UNIQUEMENT : aucun ordre réel n'est envoyé.

st.set_page_config(page_title="Simulateur boursier – 500 €", layout="wide")

CAPITAL_INITIAL = 500.0
FRAIS_TAUX = 0.001
GLISSEMENT_TAUX = 0.0005
NB_MAX_POSITIONS = 8
ORDRE_MINIMUM = 30.0
BENEFICE_NET_MIN = 0.50
BENEFICE_NET_PCT = 0.006
STOP_LOSS = 0.045
TRAILING_STOP = 0.035
SEUIL_ACHAT = 70

ACTIONS = {
    "AI.PA": "Air Liquide",
    "SAF.PA": "Safran",
    "OR.PA": "L'Oréal",
    "MC.PA": "LVMH",
    "SU.PA": "Schneider Electric",
    "TTE.PA": "TotalEnergies",
    "DG.PA": "Vinci",
    "AIR.PA": "Airbus",
    "CAP.PA": "Capgemini",
    "SAN.PA": "Sanofi",
    "BNP.PA": "BNP Paribas",
    "HO.PA": "Thales",
    "VIE.PA": "Veolia",
    "DSY.PA": "Dassault Systèmes",
    "GLE.PA": "Société Générale",
    "ASML.AS": "ASML",
    "SAP.DE": "SAP",
    "SIE.DE": "Siemens",
    "ALV.DE": "Allianz",
    "DTE.DE": "Deutsche Telekom",
    "BAS.DE": "BASF",
    "IBE.MC": "Iberdrola",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "AVGO": "Broadcom",
    "JPM": "JPMorgan",
    "COST": "Costco",
}

ACTIONS_US = {"NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "JPM", "COST"}


def initialiser():
    if "cash" not in st.session_state:
        st.session_state.cash = CAPITAL_INITIAL
        st.session_state.positions = {}
        st.session_state.historique = []
        st.session_state.derniere_action = {}


def rsi(serie, periode=14):
    delta = serie.diff()
    hausse = delta.clip(lower=0).ewm(
        alpha=1 / periode, adjust=False
    ).mean()
    baisse = (-delta.clip(upper=0)).ewm(
        alpha=1 / periode, adjust=False
    ).mean()
    rs = hausse / baisse.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def extraire_titre(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        for niveau in range(data.columns.nlevels):
            if ticker in data.columns.get_level_values(niveau):
                try:
                    resultat = data.xs(
                        ticker, axis=1, level=niveau
                    )
                    if isinstance(resultat, pd.DataFrame):
                        return resultat.dropna(how="all")
                except Exception:
                    pass
        return pd.DataFrame()

    return data.dropna(how="all")


@st.cache_data(ttl=900, show_spinner=False)
def telecharger_donnees():
    tickers = list(ACTIONS.keys())

    quotidien = yf.download(
        tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True
    )

    horaire = yf.download(
        tickers,
        period="60d",
        interval="1h",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True
    )

    change = yf.download(
        "EURUSD=X",
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    taux = None
    if change is not None and not change.empty:
        valeurs = change["Close"].dropna()
        if not valeurs.empty:
            taux = float(valeurs.iloc[-1])

    return quotidien, horaire, taux


def en_euros(ticker, prix, taux):
    if ticker in ACTIONS_US:
        if taux is None or taux <= 0:
            return None
        return prix / taux
    return prix


def analyser(ticker, quotidien, horaire, taux):
    jour = extraire_titre(quotidien, ticker)
    heure = extraire_titre(horaire, ticker)

    if jour.empty or heure.empty:
        return None

    if "Close" not in jour or "Close" not in heure:
        return None

    cj = jour["Close"].dropna()
    ch = heure["Close"].dropna()

    if len(cj) < 100 or len(ch) < 30:
        return None

    dernier_jour = float(cj.iloc[-1])
    sma20 = cj.rolling(20).mean().iloc[-1]
    sma50 = cj.rolling(50).mean().iloc[-1]
    sma100 = cj.rolling(100).mean().iloc[-1]

    dernier = float(ch.iloc[-1])
    precedent = float(ch.iloc[-2])
    moyenne5 = float(ch.rolling(5).mean().iloc[-1])
    rsi_actuel = float(rsi(ch).iloc[-1])

    prix_euro = en_euros(ticker, dernier, taux)
    if prix_euro is None:
        return None

    score = 0

    tendance = sma50 > sma100 and dernier_jour > sma50 * 0.94
    if tendance:
        score += 35

    correction = (
        dernier_jour < sma20 * 1.015
        and dernier_jour > sma50 * 0.94
    )
    if correction:
        score += 25

    reprise = dernier > moyenne5 and dernier > precedent
    if reprise:
        score += 25

    momentum = 40 <= rsi_actuel <= 68
    if momentum:
        score += 15

    return {
        "nom": ACTIONS[ticker],
        "prix": prix_euro,
        "score": score,
        "rsi": rsi_actuel,
        "tendance": tendance,
        "reprise": reprise,
        "sous_moyenne": dernier < moyenne5,
    }


def commission(montant):
    return montant * FRAIS_TAUX


def acheter(ticker, a):
    cash = st.session_state.cash

    places = NB_MAX_POSITIONS - len(
        st.session_state.positions
    )

    if places <= 0:
        return

    budget = min(cash / places, 80.0)

    if budget < ORDRE_MINIMUM:
        return

    prix = a["prix"] * (1 + GLISSEMENT_TAUX)
    frais_achat = commission(budget)

    quantite = (budget - frais_achat) / prix
    cout = quantite * prix + frais_achat

    if quantite <= 0 or cout > cash:
        return

    st.session_state.cash -= cout

    st.session_state.positions[ticker] = {
        "nom": a["nom"],
        "quantite": quantite,
        "prix_achat": prix,
        "cout": cout,
        "plus_haut": a["prix"],
        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }

    st.session_state.historique.insert(0, {
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Opération": "ACHAT",
        "Société": a["nom"],
        "Prix (€)": round(a["prix"], 2),
        "Résultat net (€)": "",
        "Motif": f"Signal d'achat : {a['score']}/100",
    })


def vendre(ticker, a, motif):
    p = st.session_state.positions[ticker]

    prix = a["prix"] * (1 - GLISSEMENT_TAUX)
    montant = p["quantite"] * prix
    montant_net = montant - commission(montant)

    resultat = montant_net - p["cout"]

    st.session_state.cash += montant_net

    st.session_state.historique.insert(0, {
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Opération": "VENTE",
        "Société": p["nom"],
        "Prix (€)": round(a["prix"], 2),
        "Résultat net (€)": round(resultat, 2),
        "Motif": motif,
    })

    del st.session_state.positions[ticker]


def gerer_positions(analyses):
    for ticker in list(st.session_state.positions.keys()):
        a = analyses.get(ticker)

        if not a:
            continue

        p = st.session_state.positions[ticker]
        p["plus_haut"] = max(p["plus_haut"], a["prix"])

        prix_sortie = a["prix"] * (1 - GLISSEMENT_TAUX)
        montant = p["quantite"] * prix_sortie
        gain_net = montant - commission(montant) - p["cout"]

        seuil = max(
            BENEFICE_NET_MIN,
            p["cout"] * BENEFICE_NET_PCT
        )

        rendement = a["prix"] / p["prix_achat"] - 1
        recul = a["prix"] / p["plus_haut"] - 1

        motif = None

        if gain_net >= seuil:
            motif = f"Petit bénéfice net atteint : {gain_net:+.2f} €"

        elif rendement <= -STOP_LOSS:
            motif = "Protection contre une baisse importante"

        elif recul <= -TRAILING_STOP and rendement > 0:
            motif = "Sécurisation du gain"

        elif not a["tendance"] and a["sous_moyenne"]:
            if gain_net > -1:
                motif = "Tendance devenue défavorable"

        if motif:
            vendre(ticker, a, motif)


def chercher_achats(analyses):
    candidats = []

    for ticker, a in analyses.items():
        if ticker in st.session_state.positions:
            continue

        if a["score"] >= SEUIL_ACHAT:
            candidats.append((ticker, a))

    candidats.sort(
        key=lambda x: x[1]["score"],
        reverse=True
    )

    for ticker, a in candidats:
        if len(st.session_state.positions) >= NB_MAX_POSITIONS:
            break

        if st.session_state.cash < ORDRE_MINIMUM:
            break

        aujourd_hui = datetime.now().strftime("%Y-%m-%d")

        if st.session_state.derniere_action.get(ticker) == aujourd_hui:
            continue

        avant = len(st.session_state.positions)
        acheter(ticker, a)

        if len(st.session_state.positions) > avant:
            st.session_state.derniere_action[ticker] = aujourd_hui


initialiser()

# Actualisation toutes les deux heures environ.
st_autorefresh(
    interval=2 * 60 * 60 * 1000,
    key="actualisation"
)

st.title("📈 Simulateur boursier – 500 €")
st.caption(
    "Simulation fictive : aucun compte bancaire ou courtier connecté."
)

with st.sidebar:
    st.subheader("Fonctionnement")
    st.write("Capital initial : **500 €**")
    st.write("Frais estimés : **0,10 % par opération**")
    st.write("Actions surveillées : **30**")
    st.write("Positions simultanées : **8 maximum**")
    st.write("Actualisation : **toutes les deux heures**")

    if st.button("Recommencer la simulation"):
        for cle in [
            "cash",
            "positions",
            "historique",
            "derniere_action"
        ]:
            st.session_state.pop(cle, None)

        initialiser()
        st.rerun()


with st.spinner("Analyse des 30 sociétés en cours..."):
    try:
        quotidien, horaire, taux = telecharger_donnees()

        analyses = {}

        for ticker in ACTIONS:
            resultat = analyser(
                ticker,
                quotidien,
                horaire,
                taux
            )

            if resultat:
                analyses[ticker] = resultat

        gerer_positions(analyses)
        chercher_achats(analyses)

    except Exception as erreur:
        st.error(
            f"Erreur lors de la récupération des cours : {erreur}"
        )
        analyses = {}


valeur_actions = 0
positions_affichees = []

for ticker, p in st.session_state.positions.items():
    a = analyses.get(ticker)

    prix = a["prix"] if a else p["prix_achat"]
    valeur = p["quantite"] * prix
    gain = valeur - p["cout"]

    valeur_actions += valeur

    positions_affichees.append({
        "Société": p["nom"],
        "Investissement (€)": round(p["cout"], 2),
        "Valeur actuelle (€)": round(valeur, 2),
        "Gain latent (€)": round(gain, 2),
        "Achetée le": p["date"],
    })


patrimoine = st.session_state.cash + valeur_actions
resultat_global = patrimoine - CAPITAL_INITIAL

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Liquidités",
    f"{st.session_state.cash:.2f} €"
)

c2.metric(
    "Valeur des actions",
    f"{valeur_actions:.2f} €"
)

c3.metric(
    "Capital total",
    f"{patrimoine:.2f} €"
)

c4.metric(
    "Résultat global",
    f"{resultat_global:+.2f} €"
)


st.divider()

gauche, droite = st.columns([1.2, 1])

with gauche:
    st.subheader("Positions ouvertes")

    if positions_affichees:
        st.dataframe(
            pd.DataFrame(positions_affichees),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "Aucune position ouverte. "
            "Le simulateur attend une occasion intéressante."
        )

with droite:
    st.subheader("Stratégie")

    st.markdown(
        """
        1. Surveillance de 30 grandes entreprises.
        2. Recherche d'une tendance de fond favorable.
        3. Attente d'une correction puis d'un signe de reprise.
        4. Achat progressif des meilleures occasions.
        5. Vente dès qu'un petit bénéfice net est atteint.
        6. Protection du capital en cas de baisse.
        """
    )


st.divider()
st.subheader("Actions intéressantes à surveiller")

if analyses:
    tableau = []

    for ticker, a in analyses.items():
        if ticker not in st.session_state.positions:
            tableau.append({
                "Société": a["nom"],
                "Cours (€)": round(a["prix"], 2),
                "Score": a["score"],
                "RSI": round(a["rsi"], 1),
                "Signal": (
                    "Occasion possible"
                    if a["score"] >= SEUIL_ACHAT
                    else "Attendre"
                ),
            })

    tableau.sort(
        key=lambda x: x["Score"],
        reverse=True
    )

    st.dataframe(
        pd.DataFrame(tableau),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "Les données ne sont pas disponibles pour le moment."
    )


st.divider()
st.subheader("Historique des opérations")

if st.session_state.historique:
    st.dataframe(
        pd.DataFrame(st.session_state.historique),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Les achats et ventes simulés apparaîtront ici.")


st.caption(
    "Attention : les signaux sont des indicateurs techniques simples, "
    "pas une analyse complète des comptes des entreprises. "
    "Les données Yahoo Finance peuvent être retardées. "
    "Les frais et le glissement sont des estimations. "
    "Les positions et l'historique sont conservés dans la session "
    "et peuvent être perdus lors d'un redémarrage de l'application. "
    "Ce programme ne garantit aucun bénéfice."
)
