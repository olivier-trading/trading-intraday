
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(
    page_title="Simulateur actions solides",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# 1. UNIVERS DES 20 ENTREPRISES
# =========================================================

UNIVERS = {
    "AIR.PA": "Airbus",
    "OR.PA": "L'Oreal",
    "MC.PA": "LVMH",
    "RMS.PA": "Hermes",
    "TTE.PA": "TotalEnergies",
    "SU.PA": "Schneider Electric",
    "AI.PA": "Air Liquide",
    "SAN.PA": "Sanofi",
    "BNP.PA": "BNP Paribas",
    "DG.PA": "Vinci",
    "CAP.PA": "Capgemini",
    "SAF.PA": "Safran",
    "DSY.PA": "Dassault Systemes",
    "EL.PA": "EssilorLuxottica",
    "ENGI.PA": "Engie",
    "RI.PA": "Pernod Ricard",
    "ASML.AS": "ASML",
    "SAP.DE": "SAP",
    "SIE.DE": "Siemens",
    "ALV.DE": "Allianz"
}

BENCHMARK = "^FCHI"


# =========================================================
# 2. PARAMETRES
# =========================================================

st.title("Simulateur de stratégie actions")

st.write(
    "Sélection d'entreprises solides, conservation des positions "
    "jusqu'à un bénéfice net après frais, puis réinvestissement."
)

with st.sidebar:
    st.header("Paramètres")

    capital_initial = st.number_input(
        "Capital initial (€)",
        min_value=100.0,
        value=500.0,
        step=50.0
    )

    frais_achat = st.number_input(
        "Frais par achat (€)",
        min_value=0.0,
        value=1.0,
        step=0.25
    )

    frais_vente = st.number_input(
        "Frais par vente (€)",
        min_value=0.0,
        value=1.0,
        step=0.25
    )

    benefice_min = st.number_input(
        "Bénéfice net minimum par vente (€)",
        min_value=0.0,
        value=0.10,
        step=0.05
    )

    nb_positions = st.slider(
        "Nombre maximal d'actions détenues",
        1, 10, 5
    )

    duree = st.slider(
        "Durée du backtest (années)",
        3, 15, 10
    )

    date_fin = st.date_input(
        "Date de fin",
        value=date.today()
    )

    lancer = st.button(
        "LANCER LE BACKTEST",
        type="primary",
        use_container_width=True
    )


# =========================================================
# 3. TELECHARGEMENT DES COURS
# =========================================================

@st.cache_data(ttl=3600)
def telecharger_cours(tickers, debut, fin):

    donnees = yf.download(
        list(tickers),
        start=debut,
        end=fin,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )

    if donnees is None or donnees.empty:
        raise ValueError("Aucun cours reçu de Yahoo Finance.")

    resultats = {}

    if isinstance(donnees.columns, pd.MultiIndex):

        for ticker in tickers:

            if ticker in donnees.columns.get_level_values(0):
                bloc = donnees[ticker]

            elif ticker in donnees.columns.get_level_values(1):
                bloc = donnees.xs(
                    ticker,
                    axis=1,
                    level=1
                )

            else:
                continue

            if "Close" in bloc.columns:
                resultats[ticker] = bloc["Close"]

    else:

        if "Close" in donnees.columns and len(tickers) == 1:
            resultats[tickers[0]] = donnees["Close"]

    if not resultats:
        raise ValueError(
            "Impossible de récupérer les cours de clôture."
        )

    prix = pd.concat(resultats, axis=1)
    prix.columns = list(resultats.keys())

    prix = prix.sort_index()
    prix = prix[~prix.index.duplicated(keep="last")]
    prix = prix.replace([np.inf, -np.inf], np.nan)

    return prix


# =========================================================
# 4. CLASSEMENT DES ENTREPRISES
# =========================================================

def calculer_scores(prix):

    rendement_3_mois = prix / prix.shift(63) - 1
    rendement_6_mois = prix / prix.shift(126) - 1

    moyenne_200 = prix.rolling(200).mean()

    volatilite = (
        prix.pct_change()
        .rolling(20)
        .std()
        * np.sqrt(20)
    )

    score = (
        0.45 * rendement_3_mois
        + 0.55 * rendement_6_mois
        - 0.25 * volatilite
    )

    # On privilégie les actions en tendance positive.
    score = score.where(prix > moyenne_200)
    score = score.where(rendement_3_mois > 0)
    score = score.where(rendement_6_mois > 0)

    # Décalage d'une séance pour éviter d'utiliser le futur.
    return score.shift(1)


# =========================================================
# 5. MOTEUR DU BACKTEST
# =========================================================

def executer_backtest(
    prix,
    capital,
    frais_buy,
    frais_sell,
    marge,
    max_positions
):

    scores = calculer_scores(prix)

    cash = float(capital)
    positions = {}

    transactions = []
    historique = []

    debut = 210

    if len(prix) <= debut:
        raise ValueError(
            "Pas assez de données historiques."
        )

    for i in range(debut, len(prix)):

        jour = prix.index[i]
        cours_jour = prix.iloc[i]
        score_jour = scores.iloc[i]

        # -------------------------------------------------
        # A. VENTES : UNIQUEMENT SI BENEFICE NET SUFFISANT
        # -------------------------------------------------

        ventes_du_jour = []

        for ticker, position in list(positions.items()):

            cours = cours_jour.get(ticker, np.nan)

            if pd.isna(cours) or cours <= 0:
                continue

            montant_vente = (
                position["quantite"] * float(cours)
                - frais_sell
            )

            benefice_net = (
                montant_vente
                - position["cout_total"]
            )

            if benefice_net >= marge:
                ventes_du_jour.append(
                    (ticker, float(cours), montant_vente,
                     benefice_net)
                )

        titres_vendus_aujourdhui = set()

        for ticker, cours, montant_vente, benefice in ventes_du_jour:

            position = positions.pop(ticker)

            cash += montant_vente
            titres_vendus_aujourdhui.add(ticker)

            transactions.append({
                "Date": jour,
                "Type": "VENTE",
                "Ticker": ticker,
                "Entreprise": UNIVERS.get(ticker, ticker),
                "Cours": cours,
                "Quantite": position["quantite"],
                "Frais": frais_sell,
                "Benefice net": benefice,
                "Tresorerie": cash
            })

        # -------------------------------------------------
        # B. NOUVEAUX ACHATS APRES RECLASSEMENT
        # -------------------------------------------------

        while len(positions) < max_positions:

            places_libres = max_positions - len(positions)

            if places_libres <= 0:
                break

            # Répartition de la trésorerie entre les places libres.
            budget = cash / places_libres

            if budget <= frais_buy:
                break

            candidats = score_jour.dropna()

            candidats = candidats[
                ~candidats.index.isin(positions.keys())
            ]

            candidats = candidats[
                ~candidats.index.isin(titres_vendus_aujourdhui)
            ]

            candidats = candidats.sort_values(
                ascending=False
            )

            if candidats.empty:
                break

            achat_effectue = False

            for ticker, score in candidats.items():

                cours = cours_jour.get(ticker, np.nan)

                if pd.isna(cours) or cours <= 0:
                    continue

                montant_depense = min(budget, cash)
                montant_investi = montant_depense - frais_buy

                if montant_investi <= 0:
                    continue

                quantite = montant_investi / float(cours)

                positions[ticker] = {
                    "quantite": quantite,
                    "cout_total": montant_depense,
                    "date_achat": jour,
                    "cours_achat": float(cours)
                }

                cash -= montant_depense

                transactions.append({
                    "Date": jour,
                    "Type": "ACHAT",
                    "Ticker": ticker,
                    "Entreprise": UNIVERS.get(ticker, ticker),
                    "Cours": float(cours),
                    "Quantite": quantite,
                    "Frais": frais_buy,
                    "Benefice net": np.nan,
                    "Tresorerie": cash
                })

                achat_effectue = True
                break

            if not achat_effectue:
                break

        # -------------------------------------------------
        # C. VALORISATION DU PORTEFEUILLE
        # -------------------------------------------------

        valeur_actions = 0.0

        for ticker, position in positions.items():

            cours = cours_jour.get(ticker, np.nan)

            if pd.isna(cours):
                continue

            valeur_actions += (
                position["quantite"] * float(cours)
            )

        valeur_totale = cash + valeur_actions

        historique.append({
            "Date": jour,
            "Capital": valeur_totale,
            "Tresorerie": cash,
            "Valeur actions": valeur_actions,
            "Nombre positions": len(positions)
        })

    # =====================================================
    # 6. RESULTATS
    # =====================================================

    courbe = pd.DataFrame(historique).set_index("Date")

    transactions_df = pd.DataFrame(transactions)

    valeur_finale = float(courbe["Capital"].iloc[-1])

    gain = valeur_finale - capital

    performance = valeur_finale / capital - 1

    jours = max(
        (courbe.index[-1] - courbe.index[0]).days,
        1
    )

    rendement_annuel = (
        (valeur_finale / capital) ** (365.25 / jours) - 1
        if valeur_finale > 0 else -1
    )

    sommet = courbe["Capital"].cummax()

    drawdown = courbe["Capital"] / sommet - 1

    perte_max = float(drawdown.min())

    total_frais = (
        transactions_df["Frais"].fillna(0).sum()
        if not transactions_df.empty else 0
    )

    achats = (
        transactions_df[transactions_df["Type"] == "ACHAT"]
        if not transactions_df.empty else pd.DataFrame()
    )

    ventes = (
        transactions_df[transactions_df["Type"] == "VENTE"]
        if not transactions_df.empty else pd.DataFrame()
    )

    benefices_realises = (
        ventes["Benefice net"].sum()
        if not ventes.empty else 0
    )

    # -----------------------------------------------------
    # POSITIONS ENCORE DETENUES A LA FIN
    # -----------------------------------------------------

    positions_finales = []

    for ticker, position in positions.items():

        cours = prix[ticker].dropna().iloc[-1]

        valeur = position["quantite"] * float(cours)

        gain_latent = (
            valeur
            - frais_sell
            - position["cout_total"]
        )

        seuil_vente = (
            position["cout_total"]
            + frais_sell
            + marge
        ) / position["quantite"]

        positions_finales.append({
            "Entreprise": UNIVERS.get(ticker, ticker),
            "Ticker": ticker,
            "Date achat": position["date_achat"].date(),
            "Prix achat": position["cours_achat"],
            "Dernier cours": float(cours),
            "Seuil vente": seuil_vente,
            "Gain latent net": gain_latent
        })

    return {
        "courbe": courbe,
        "transactions": transactions_df,
        "positions": pd.DataFrame(positions_finales),
        "valeur_finale": valeur_finale,
        "gain": gain,
        "performance": performance,
        "rendement_annuel": rendement_annuel,
        "perte_max": perte_max,
        "total_frais": total_frais,
        "nb_achats": len(achats),
        "nb_ventes": len(ventes),
        "benefices_realises": benefices_realises
    }


# =========================================================
# 7. EXECUTION ET AFFICHAGE
# =========================================================

if lancer:

    fin = pd.Timestamp(date_fin)

    debut = (
        fin
        - pd.DateOffset(years=duree)
        - pd.Timedelta(days=400)
    )

    fin_telechargement = (
        fin + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:

        with st.spinner(
            "Téléchargement des cours et calcul du backtest..."
        ):

            prix = telecharger_cours(
                tuple(UNIVERS.keys()),
                debut.strftime("%Y-%m-%d"),
                fin_telechargement
            )

            prix = prix.dropna(
                axis=1,
                thresh=max(250, int(len(prix) * 0.55))
            )

            if prix.shape[1] < 5:
                raise ValueError(
                    "Moins de cinq actions disposent de suffisamment de données."
                )

            resultat = executer_backtest(
                prix,
                capital_initial,
                frais_achat,
                frais_vente,
                benefice_min,
                nb_positions
            )

        st.success("Backtest terminé.")

        st.subheader("Résultats principaux")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Capital final",
            f'{resultat["valeur_finale"]:.2f} €'
        )

        c2.metric(
            "Gain / perte",
            f'{resultat["gain"]:.2f} €'
        )

        c3.metric(
            "Performance totale",
            f'{resultat["performance"]:.2%}'
        )

        c4.metric(
            "Rendement annualisé",
            f'{resultat["rendement_annuel"]:.2%}'
        )

        c5, c6, c7, c8 = st.columns(4)

        c5.metric(
            "Perte maximale",
            f'{resultat["perte_max"]:.2%}'
        )

        c6.metric(
            "Frais cumulés",
            f'{resultat["total_frais"]:.2f} €'
        )

        c7.metric(
            "Achats / ventes",
            f'{resultat["nb_achats"]} / {resultat["nb_ventes"]}'
        )

        c8.metric(
            "Bénéfices réalisés",
            f'{resultat["benefices_realises"]:.2f} €'
        )

        # -------------------------------------------------
        # COURBE DU CAPITAL
        # -------------------------------------------------

        st.subheader("Evolution du capital")

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(
            resultat["courbe"].index,
            resultat["courbe"]["Capital"],
            label="Stratégie"
        )

        ax.set_ylabel("Capital en euros")
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

        # -------------------------------------------------
        # TRESORERIE ET ACTIONS
        # -------------------------------------------------

        derniere_ligne = resultat["courbe"].iloc[-1]

        c1, c2 = st.columns(2)

        c1.metric(
            "Trésorerie disponible",
            f'{derniere_ligne["Tresorerie"]:.2f} €'
        )

        c2.metric(
            "Valeur des actions",
            f'{derniere_ligne["Valeur actions"]:.2f} €'
        )

        # -------------------------------------------------
        # POSITIONS OUVERTES
        # -------------------------------------------------

        st.subheader("Actions encore détenues")

        if resultat["positions"].empty:

            st.info("Aucune action détenue à la fin du backtest.")

        else:

            st.dataframe(
                resultat["positions"],
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # HISTORIQUE DES OPERATIONS
        # -------------------------------------------------

        st.subheader("Historique des achats et ventes")

        if resultat["transactions"].empty:

            st.info("Aucune opération réalisée.")

        else:

            st.dataframe(
                resultat["transactions"].sort_values(
                    "Date",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            csv = resultat["transactions"].to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "Télécharger les opérations en CSV",
                csv,
                "historique_backtest.csv",
                "text/csv"
            )

        st.warning(
            "Attention : le modèle ne vend jamais une action à perte. "
            "Une position peut donc rester immobilisée longtemps ou perdre "
            "une grande partie de sa valeur. Les fractions d'actions sont "
            "autorisées. Les frais sont fixes, mais les impôts, les spreads "
            "et le slippage ne sont pas intégrés. L'univers actuel crée "
            "également un biais de survivance dans le backtest historique."
        )

    except Exception as erreur:

        st.error(
            f"Erreur pendant le backtest : {erreur}"
        )

else:

    st.info(
        "Choisis tes paramètres à gauche, puis clique sur "
        "LANCER LE BACKTEST."
    )

    st.subheader("Les 20 entreprises suivies")

    liste = pd.DataFrame([
        {"Ticker": ticker, "Entreprise": nom}
        for ticker, nom in UNIVERS.items()
    ])

    st.dataframe(
        liste,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    **Principe de fonctionnement :**

    1. Le programme analyse les tendances sur trois et six mois.
    2. Il privilégie les actions au-dessus de leur moyenne mobile à 200 jours.
    3. Il achète les meilleures candidates disponibles.
    4. Il conserve chaque action jusqu'à ce que sa vente couvre les frais d'achat, les frais de vente et le bénéfice minimum demandé.
    5. Après une vente, il rescane les entreprises et réinvestit la trésorerie disponible.
    """)
