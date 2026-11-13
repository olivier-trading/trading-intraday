
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# SIMULATEUR HISTORIQUE - 500 € - 10 ANS
# Stratégie : sélection quotidienne, arbitrages hebdomadaires,
# conservation jusqu'au rang 15, filtre de tendance CAC 40.
# AUCUN ordre réel n'est envoyé.
# ============================================================

st.set_page_config(
    page_title="Backtest actions - stratégie optimisée",
    layout="wide"
)

CAPITAL_INITIAL = 500.0
FRAIS = 0.001                 # 0,10 % par achat ou vente
NB_ACTIONS = 10
RANG_SORTIE = 15
JOURS_ENTRE_ARBITRAGES = 5
PERIODE = "10y"

ACTIONS = [
    "AIR.PA", "OR.PA", "SAN.PA", "SU.PA", "BNP.PA",
    "DG.PA", "AI.PA", "CAP.PA", "VIE.PA", "ORA.PA",
    "ENGI.PA", "ACA.PA", "RNO.PA", "STLAM.PA", "GLE.PA",
    "HO.PA", "RI.PA", "MC.PA", "KER.PA", "PUB.PA"
]

CAC40 = "^FCHI"


# ------------------------------------------------------------
# EXTRACTION ROBUSTE DES COURS YAHOO FINANCE
# ------------------------------------------------------------

def extraire_clotures(data):
    if data is None or data.empty:
        raise ValueError("Aucune donnée reçue de Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        niveaux = [
            list(data.columns.get_level_values(0)),
            list(data.columns.get_level_values(1))
        ]

        if "Close" in niveaux[0]:
            close = data["Close"]
        elif "Close" in niveaux[1]:
            close = data.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Colonne Close introuvable dans les données.")

    else:
        if "Close" not in data.columns:
            raise ValueError("Colonne Close introuvable.")
        close = data["Close"]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.copy()
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan)

    return close


@st.cache_data(ttl=3600, show_spinner=False)
def telecharger_donnees():
    donnees_actions = yf.download(
        ACTIONS,
        period=PERIODE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True
    )

    cours = extraire_clotures(donnees_actions)
    cours = cours.reindex(columns=ACTIONS)
    cours = cours.dropna(axis=1, how="all")
    cours = cours.sort_index()

    donnees_cac = yf.download(
        CAC40,
        period=PERIODE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    cac = extraire_clotures(donnees_cac)
    cac = cac.iloc[:, 0].rename("CAC40")

    return cours, cac


# ------------------------------------------------------------
# INDICATEURS DE SELECTION
# ------------------------------------------------------------

def calculer_scores(cours):
    rendements = cours.pct_change(fill_method=None)

    momentum_20 = cours / cours.shift(20) - 1
    momentum_120 = cours / cours.shift(120) - 1

    volatilite_20 = (
        rendements
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # Même formule de sélection que dans le backtest précédent
    scores = (
        0.60 * momentum_20
        + 0.40 * momentum_120
        - 0.50 * volatilite_20
    )

    return scores


# ------------------------------------------------------------
# CALCUL DES INDICATEURS DE PERFORMANCE
# ------------------------------------------------------------

def calculer_indicateurs(capital):
    capital = capital.dropna()

    if len(capital) < 2:
        return {}

    rendement_total = capital.iloc[-1] / CAPITAL_INITIAL - 1

    duree_annees = (
        (capital.index[-1] - capital.index[0]).days / 365.25
    )

    if duree_annees > 0 and capital.iloc[-1] > 0:
        rendement_annuel = (
            (capital.iloc[-1] / CAPITAL_INITIAL)
            ** (1 / duree_annees) - 1
        )
    else:
        rendement_annuel = np.nan

    drawdown = capital / capital.cummax() - 1
    perte_max = drawdown.min()

    rendements_journaliers = capital.pct_change().dropna()

    volatilite = (
        rendements_journaliers.std() * np.sqrt(252)
        if len(rendements_journaliers) > 1 else np.nan
    )

    rendement_moyen = rendements_journaliers.mean() * 252
    ratio_sharpe = (
        rendement_moyen / volatilite
        if pd.notna(volatilite) and volatilite > 0 else np.nan
    )

    return {
        "capital_final": capital.iloc[-1],
        "gain": capital.iloc[-1] - CAPITAL_INITIAL,
        "rendement_total": rendement_total,
        "rendement_annuel": rendement_annuel,
        "perte_max": perte_max,
        "volatilite": volatilite,
        "sharpe": ratio_sharpe
    }


# ------------------------------------------------------------
# BACKTEST AVEC ARBITRAGES LIMITES
# ------------------------------------------------------------

def lancer_backtest(cours, cac, filtre_tendance=True):
    cours = cours.copy().sort_index()
    cac = cac.copy().sort_index()

    # On garde uniquement les actions disposant de suffisamment
    # de données pour calculer les indicateurs.
    scores = calculer_scores(cours)

    cac_aligne = cac.reindex(cours.index).ffill()
    moyenne_cac_200 = cac_aligne.rolling(200).mean()

    # Il faut au moins 120 jours d'historique.
    debut = 121

    if filtre_tendance:
        debut = max(debut, 201)

    if len(cours) <= debut + 5:
        raise ValueError("Historique insuffisant pour le backtest.")

    tresorerie = CAPITAL_INITIAL
    portefeuille = {}  # ticker -> nombre d'actions détenues

    frais_cumules = 0.0
    nombre_achats = 0
    nombre_ventes = 0

    historique_capital = []
    historique_tresorerie = []
    historique_positions = []

    dates = cours.index

    for i in range(debut, len(dates)):
        date = dates[i]

        # Cours de clôture du jour
        prix_du_jour = cours.loc[date]

        # Valeur du portefeuille avant les arbitrages du jour
        valeur_positions = 0.0

        for ticker, quantite in portefeuille.items():
            prix = prix_du_jour.get(ticker, np.nan)
            if pd.notna(prix) and prix > 0:
                valeur_positions += quantite * float(prix)

        valeur_totale = tresorerie + valeur_positions

        # Arbitrage seulement tous les 5 jours de bourse.
        jour_arbitrage = ((i - debut) % JOURS_ENTRE_ARBITRAGES == 0)

        if jour_arbitrage:
            scores_du_jour = scores.loc[date].dropna()
            classement = scores_du_jour.sort_values(
                ascending=False
            )

            top_10 = list(classement.head(NB_ACTIONS).index)
            rangs = {
                ticker: rang
                for rang, ticker in enumerate(classement.index, start=1)
            }

            # Filtre de tendance : CAC 40 au-dessus de sa moyenne
            # mobile 200 jours.
            marche_favorable = True

            if filtre_tendance:
                niveau_cac = cac_aligne.loc[date]
                moyenne = moyenne_cac_200.loc[date]

                marche_favorable = (
                    pd.notna(niveau_cac)
                    and pd.notna(moyenne)
                    and niveau_cac >= moyenne
                )

            if not marche_favorable:
                # Passage en liquidités lorsque le filtre est négatif.
                cibles = []
            else:
                # On conserve les actions déjà détenues si elles
                # restent dans les 15 premières.
                conservees = []

                for ticker in portefeuille:
                    if rangs.get(ticker, 999) <= RANG_SORTIE:
                        conservees.append(ticker)

                # On complète avec les meilleures actions disponibles.
                cibles = conservees.copy()

                for ticker in top_10:
                    if ticker not in cibles and len(cibles) < NB_ACTIONS:
                        cibles.append(ticker)

            # VENTES : uniquement des actions qui ne sont plus retenues.
            a_vendre = [
                ticker for ticker in portefeuille
                if ticker not in cibles
            ]

            for ticker in a_vendre:
                prix = prix_du_jour.get(ticker, np.nan)

                if pd.notna(prix) and prix > 0:
                    montant = portefeuille[ticker] * float(prix)
                    commission = montant * FRAIS

                    tresorerie += montant - commission
                    frais_cumules += commission
                    nombre_ventes += 1

                del portefeuille[ticker]

            # ACHATS : on utilise la trésorerie disponible pour
            # remplir les places vacantes, sans vendre les gagnantes
            # pour rééquilibrer artificiellement les poids.
            nouvelles = [
                ticker for ticker in cibles
                if ticker not in portefeuille
            ]

            for position in range(len(nouvelles)):
                ticker = nouvelles[position]
                places_restantes = len(nouvelles) - position

                prix = prix_du_jour.get(ticker, np.nan)

                if pd.isna(prix) or prix <= 0 or tresorerie <= 0:
                    continue

                budget = tresorerie / places_restantes
                montant_achat = budget / (1 + FRAIS)
                commission = montant_achat * FRAIS

                quantite = montant_achat / float(prix)
                cout_total = montant_achat + commission

                if cout_total <= tresorerie:
                    portefeuille[ticker] = quantite
                    tresorerie -= cout_total
                    frais_cumules += commission
                    nombre_achats += 1

        # Valorisation après arbitrages et commissions.
        valeur_positions = 0.0

        for ticker, quantite in portefeuille.items():
            prix = prix_du_jour.get(ticker, np.nan)

            if pd.notna(prix) and prix > 0:
                valeur_positions += quantite * float(prix)

        valeur_totale = tresorerie + valeur_positions

        historique_capital.append((date, valeur_totale))
        historique_tresorerie.append((date, tresorerie))
        historique_positions.append((date, len(portefeuille)))

    capital = pd.Series(
        dict(historique_capital),
        name="Stratégie optimisée"
    )

    tresorerie_historique = pd.Series(
        dict(historique_tresorerie),
        name="Trésorerie"
    )

    positions_historique = pd.Series(
        dict(historique_positions),
        name="Nombre de positions"
    )

    # Comparaison avec le CAC 40 sur les mêmes dates.
    cac_comparable = cac_aligne.reindex(capital.index).dropna()

    dates_communes = capital.index.intersection(cac_comparable.index)
    capital = capital.reindex(dates_communes)
    cac_comparable = cac_comparable.reindex(dates_communes)

    if len(cac_comparable) > 1:
        benchmark = (
            CAPITAL_INITIAL
            * cac_comparable / cac_comparable.iloc[0]
        )
        benchmark.name = "CAC 40"
    else:
        benchmark = pd.Series(dtype=float)

    return {
        "capital": capital,
        "benchmark": benchmark,
        "tresorerie": tresorerie_historique.reindex(dates_communes),
        "positions": positions_historique.reindex(dates_communes),
        "frais": frais_cumules,
        "achats": nombre_achats,
        "ventes": nombre_ventes,
        "portefeuille": portefeuille
    }


# ------------------------------------------------------------
# INTERFACE STREAMLIT
# ------------------------------------------------------------

st.title("Backtest actions - stratégie optimisée")

st.write(
    "Simulation historique sur 10 ans avec 500 € de départ. "
    "Les opérations sont fictives : aucun ordre réel n'est transmis."
)

st.sidebar.header("Paramètres")

filtre_tendance = st.sidebar.checkbox(
    "Activer le filtre de tendance CAC 40 (200 jours)",
    value=True
)

st.sidebar.write("Capital initial : 500 €")
st.sidebar.write("Nombre maximal d'actions : 10")
st.sidebar.write("Arbitrage : tous les 5 jours de bourse")
st.sidebar.write("Commission : 0,10 % par opération")

if st.button("Lancer le backtest", type="primary"):
    st.cache_data.clear()

try:
    with st.spinner("Téléchargement des cours et calcul du backtest..."):
        cours, cac = telecharger_donnees()

        if cours.shape[1] < 10:
            st.error(
                "Moins de 10 actions disposent de données suffisantes. "
                "Vérifie les données Yahoo Finance."
            )
            st.stop()

        resultat = lancer_backtest(
            cours,
            cac,
            filtre_tendance=filtre_tendance
        )

    capital = resultat["capital"]
    benchmark = resultat["benchmark"]

    if capital.empty:
        st.error("Le backtest n'a produit aucun résultat.")
        st.stop()

    indicateurs = calculer_indicateurs(capital)
    indicateurs_cac = calculer_indicateurs(benchmark)

    st.subheader("Résultats de la stratégie")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Capital final",
        f"{indicateurs['capital_final']:.2f} €"
    )

    c2.metric(
        "Gain net",
        f"{indicateurs['gain']:.2f} €"
    )

    c3.metric(
        "Rendement annualisé",
        f"{indicateurs['rendement_annuel'] * 100:.2f} %"
    )

    c4.metric(
        "Perte maximale",
        f"{indicateurs['perte_max'] * 100:.2f} %"
    )

    st.subheader("Comparaison avec le CAC 40")

    if not benchmark.empty:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Notre stratégie",
            f"{capital.iloc[-1]:.2f} €"
        )

        col2.metric(
            "CAC 40",
            f"{benchmark.iloc[-1]:.2f} €"
        )

        col3.metric(
            "Écart",
            f"{capital.iloc[-1] - benchmark.iloc[-1]:+.2f} €"
        )

        graphique = pd.concat(
            [capital, benchmark],
            axis=1
        ).dropna()

        st.line_chart(graphique)

    st.subheader("Évolution du portefeuille")

    st.line_chart(
        pd.DataFrame({
            "Valeur totale (€)": capital,
            "Trésorerie (€)": resultat["tresorerie"]
        })
    )

    st.subheader("Risque et frais")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Frais cumulés",
        f"{resultat['frais']:.2f} €"
    )

    col2.metric(
        "Achats",
        resultat["achats"]
    )

    col3.metric(
        "Ventes",
        resultat["ventes"]
    )

    col4.metric(
        "Ratio de Sharpe",
        f"{indicateurs['sharpe']:.2f}"
        if pd.notna(indicateurs["sharpe"]) else "N/D"
    )

    st.subheader("Nombre d'actions détenues")

    st.line_chart(resultat["positions"])

    st.subheader("Dernières positions détenues")

    dernier_jour = cours.index[-1]
    lignes = []

    for ticker, quantite in resultat["portefeuille"].items():
        prix = cours[ticker].dropna().iloc[-1]
        valeur = quantite * float(prix)

        lignes.append({
            "Action": ticker,
            "Quantité": round(quantite, 5),
            "Dernier cours": round(float(prix), 2),
            "Valeur estimée": round(valeur, 2)
        })

    if lignes:
        tableau = pd.DataFrame(lignes)
        st.dataframe(tableau, use_container_width=True)
    else:
        st.info("Le portefeuille est actuellement en liquidités.")

    st.subheader("Performance annuelle")

    annuel = capital.resample("YE").last()
    annuel = annuel.pct_change()

    if len(annuel) > 0:
        premier_rendement = (
            capital.iloc[0] / CAPITAL_INITIAL - 1
        )
        annuel.iloc[0] = premier_rendement

        tableau_annuel = (
            annuel * 100
        ).round(2).to_frame("Rendement (%)")

        tableau_annuel.index = tableau_annuel.index.year
        st.dataframe(tableau_annuel, use_container_width=True)

    st.caption(
        "Attention : les actions sélectionnées sont celles d'aujourd'hui. "
        "Le backtest présente donc un biais de survivance. "
        "Le CAC 40 est ici un indice de prix, sans réinvestissement "
        "des dividendes. Les frais de courtage sont simulés à 0,10 %, "
        "sans spread ni glissement de prix. Les fractions d'actions "
        "sont autorisées."
    )

except Exception as erreur:
    st.error(f"Erreur pendant le backtest : {erreur}")
    st.info(
        "Si l'erreur concerne les cours, réessaie dans quelques minutes. "
        "Yahoo Finance peut temporairement limiter les téléchargements."
    )
