
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Backtest Trading 10 ans",
    page_icon="📈",
    layout="wide"
)

CAPITAL_INITIAL = 500.0
NB_ACTIONS = 10
FRAIS = 0.001

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


# =========================================================
# TELECHARGEMENT DES DONNEES
# =========================================================

def extraire_clotures(data):

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):

        close = None

        for niveau in range(data.columns.nlevels):

            if "Close" in data.columns.get_level_values(niveau):

                close = data.xs(
                    "Close",
                    axis=1,
                    level=niveau
                )

                break

        if close is None:
            return pd.DataFrame()

    else:

        if "Close" not in data.columns:
            return pd.DataFrame()

        close = data["Close"]

        if isinstance(close, pd.Series):
            close = close.to_frame()

    return close


def telecharger_historique():

    fin = date.today() + timedelta(days=1)
    debut = fin - timedelta(days=365 * 10 + 10)

    tickers = list(ACTIONS.keys())

    donnees = yf.download(
        tickers,
        start=debut,
        end=fin,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True
    )

    prix = extraire_clotures(donnees)

    if prix.empty:
        raise ValueError(
            "Yahoo Finance n'a retourne aucun cours exploitable."
        )

    prix = prix.rename_axis(columns="Ticker")

    prix = prix.reindex(
        columns=tickers
    )

    prix = prix.apply(
        pd.to_numeric,
        errors="coerce"
    )

    prix = prix.sort_index()
    prix = prix[~prix.index.duplicated(keep="last")]
    prix = prix.dropna(how="all")

    benchmark_brut = yf.download(
        "^FCHI",
        start=debut,
        end=fin,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    benchmark = extraire_clotures(benchmark_brut)

    if benchmark.empty:
        raise ValueError(
            "Impossible de recuperer le CAC 40."
        )

    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]

    benchmark = pd.to_numeric(
        benchmark,
        errors="coerce"
    ).dropna()

    return prix, benchmark


# =========================================================
# CALCUL DU SCORE
# =========================================================

def calculer_scores(prix):

    rendement_20 = (
        prix / prix.shift(20) - 1
    )

    rendement_120 = (
        prix / prix.shift(120) - 1
    )

    rendements_journaliers = prix.pct_change(
        fill_method=None
    )

    volatilite = rendements_journaliers.rolling(
        window=20,
        min_periods=20
    ).std()

    score = (
        0.60 * rendement_20
        + 0.40 * rendement_120
        - 0.50 * volatilite
    )

    return score


# =========================================================
# CONSTRUCTION DU PORTEFEUILLE
# =========================================================

def construire_portefeuille(prix):

    scores = calculer_scores(prix)

    # Le signal du jour utilise uniquement les cours
    # disponibles a la cloture de la veille.
    signaux = scores.shift(1)

    poids = pd.DataFrame(
        0.0,
        index=prix.index,
        columns=prix.columns
    )

    for jour in prix.index:

        classement = signaux.loc[jour].dropna()

        # On exclut les actions sans cours disponible
        disponibles = prix.loc[jour].dropna().index

        classement = classement[
            classement.index.isin(disponibles)
        ]

        if classement.empty:
            continue

        selection = classement.nlargest(
            min(NB_ACTIONS, len(classement))
        )

        nombre = len(selection)

        poids.loc[
            jour,
            selection.index
        ] = 1.0 / nombre

    return poids, scores


# =========================================================
# BACKTEST
# =========================================================

def lancer_backtest(prix, benchmark):

    poids, scores = construire_portefeuille(prix)

    rendements = prix.pct_change(
        fill_method=None
    ).fillna(0)

    # Rotation du portefeuille
    turnover = poids.diff().abs().sum(axis=1)

    # Premier investissement : 10 achats
    if len(turnover) > 0:
        turnover.iloc[0] = poids.iloc[0].abs().sum()

    # Performance avant frais
    performance_brute = (
        poids * rendements
    ).sum(axis=1)

    # Commission sur achats et ventes
    commissions_pct = turnover * FRAIS

    performance_nette = (
        performance_brute - commissions_pct
    )

    # Evolution du capital
    capital = CAPITAL_INITIAL * (
        1 + performance_nette
    ).cumprod()

    capital = capital.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Date du premier investissement effectif
    dates_investies = poids.sum(axis=1)

    dates_investies = dates_investies[
        dates_investies > 0
    ]

    if dates_investies.empty:
        raise ValueError(
            "Pas assez de donnees pour constituer un portefeuille."
        )

    premiere_date = dates_investies.index[0]

    capital = capital.loc[premiere_date:]

    performance_nette = performance_nette.loc[
        premiere_date:
    ]

    commissions_pct = commissions_pct.loc[
        premiere_date:
    ]

    turnover = turnover.loc[premiere_date:]

    poids = poids.loc[premiere_date:]

    # Frais exprimes en euros
    capital_precedent = capital.shift(1).fillna(
        CAPITAL_INITIAL
    )

    frais_euros = (
        capital_precedent * commissions_pct
    )

    total_frais = float(frais_euros.sum())

    # Benchmark CAC 40
    benchmark_aligne = benchmark.reindex(
        capital.index
    ).ffill()

    benchmark_aligne = benchmark_aligne.dropna()

    capital = capital.loc[
        benchmark_aligne.index
    ]

    capital_initial_benchmark = float(
        benchmark_aligne.iloc[0]
    )

    valeur_benchmark = (
        CAPITAL_INITIAL
        * (1 - FRAIS)
        * benchmark_aligne
        / capital_initial_benchmark
    )

    # Drawdown
    sommet = capital.cummax()

    drawdown = capital / sommet - 1

    drawdown_max = float(drawdown.min() * 100)

    # Rendement global
    capital_final = float(capital.iloc[-1])

    gain = capital_final - CAPITAL_INITIAL

    rendement_global = (
        capital_final / CAPITAL_INITIAL - 1
    ) * 100

    # Rendement annuel moyen
    nombre_jours = (
        capital.index[-1] - capital.index[0]
    ).days

    if nombre_jours > 0 and capital_final > 0:

        rendement_annuel = (
            (capital_final / CAPITAL_INITIAL)
            ** (365.25 / nombre_jours) - 1
        ) * 100

    else:
        rendement_annuel = 0.0

    # Volatilite et ratio de Sharpe
    volatilite_annuelle = (
        performance_nette.std() * np.sqrt(252) * 100
    )

    if performance_nette.std() > 0:

        sharpe = (
            performance_nette.mean()
            / performance_nette.std()
            * np.sqrt(252)
        )

    else:
        sharpe = 0.0

    # Nombre de modifications de positions
    modifications = poids.diff().fillna(
        poids
    )

    nombre_ordres = int(
        (modifications.abs() > 0.0000001).sum().sum()
    )

    # Tableau annuel
    annuel = capital.groupby(
        capital.index.year
    ).last()

    lignes_annuelles = []

    precedent = CAPITAL_INITIAL

    for annee, valeur in annuel.items():

        rendement = (
            valeur / precedent - 1
        ) * 100

        lignes_annuelles.append({
            "Annee": int(annee),
            "Capital final (€)": round(float(valeur), 2),
            "Performance (%)": round(float(rendement), 2)
        })

        precedent = valeur

    tableau_annuel = pd.DataFrame(
        lignes_annuelles
    )

    return {
        "capital": capital,
        "benchmark": valeur_benchmark,
        "drawdown": drawdown,
        "scores": scores,
        "poids": poids,
        "capital_final": capital_final,
        "gain": gain,
        "rendement_global": rendement_global,
        "rendement_annuel": rendement_annuel,
        "drawdown_max": drawdown_max,
        "volatilite": volatilite_annuelle,
        "sharpe": sharpe,
        "frais": total_frais,
        "ordres": nombre_ordres,
        "annuel": tableau_annuel,
        "benchmark_final": float(valeur_benchmark.iloc[-1])
    }


# =========================================================
# INTERFACE STREAMLIT
# =========================================================

st.title("📈 Backtest historique - Trading sur 10 ans")

st.markdown(
    """
    Simulation d'une strategie de selection automatique
    de 10 actions parmi 20 titres europeens.

    Capital initial : **500 €**

    Periode : **10 ans**

    Frequence : **une decision par seance boursiere**
    """
)

st.divider()

if st.button(
    "🚀 LANCER LE BACKTEST SUR 10 ANS",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Telechargement des 10 annees de donnees..."
        ):

            prix, benchmark = telecharger_historique()

        with st.spinner(
            "Calcul de la strategie et des arbitrages..."
        ):

            resultat = lancer_backtest(
                prix,
                benchmark
            )

        st.session_state.resultat_backtest = resultat
        st.session_state.prix_backtest = prix

        st.success(
            "Backtest termine avec succes."
        )

    except Exception as e:

        st.error(
            f"Erreur pendant le backtest : {e}"
        )


# =========================================================
# AFFICHAGE DES RESULTATS
# =========================================================

if "resultat_backtest" in st.session_state:

    r = st.session_state.resultat_backtest

    st.divider()

    st.header("1. Resultat global")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Capital initial",
        "500 €"
    )

    c2.metric(
        "Capital final",
        f"{r['capital_final']:.2f} €"
    )

    c3.metric(
        "Gain / Perte",
        f"{r['gain']:+.2f} €"
    )

    c4.metric(
        "Performance totale",
        f"{r['rendement_global']:+.2f} %"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Rendement annualise",
        f"{r['rendement_annuel']:.2f} %"
    )

    c6.metric(
        "Perte maximale",
        f"{r['drawdown_max']:.2f} %"
    )

    c7.metric(
        "Frais cumules",
        f"{r['frais']:.2f} €"
    )

    c8.metric(
        "Nombre de modifications",
        str(r["ordres"])
    )

    st.divider()

    # Evolution du capital
    st.header("2. Evolution du capital")

    graphique = pd.DataFrame({
        "Strategie automatique": r["capital"],
        "CAC 40": r["benchmark"]
    })

    st.line_chart(
        graphique,
        use_container_width=True
    )

    st.caption(
        "Comparaison sur la periode commune, avec "
        "500 € de depart et frais d'entree de 0,10 %."
    )

    st.write(
        f"Capital final CAC 40 : "
        f"**{r['benchmark_final']:.2f} €**"
    )

    ecart = (
        r["capital_final"] - r["benchmark_final"]
    )

    st.write(
        f"Ecart strategie / CAC 40 : **{ecart:+.2f} €**"
    )

    st.divider()

    # Drawdown
    st.header("3. Risque et pertes maximales")

    graphique_risque = (
        r["drawdown"] * 100
    ).to_frame("Drawdown (%)")

    st.line_chart(
        graphique_risque,
        use_container_width=True
    )

    st.write(
        f"Drawdown maximal : **{r['drawdown_max']:.2f} %**"
    )

    st.write(
        f"Volatilite annualisee : **{r['volatilite']:.2f} %**"
    )

    st.write(
        f"Ratio de Sharpe : **{r['sharpe']:.2f}**"
    )

    st.divider()

    # Performance annuelle
    st.header("4. Performance annee par annee")

    st.dataframe(
        r["annuel"],
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        r["annuel"].set_index("Annee")[
            "Performance (%)"
        ]
    )

    st.divider()

    # Derniere selection
    st.header("5. Derniere selection des 10 actions")

    prix = st.session_state.prix_backtest

    derniere_date = r["poids"].index[-1]

    derniere_selection = r["poids"].loc[
        derniere_date
    ]

    derniere_selection = derniere_selection[
        derniere_selection > 0
    ]

    lignes = []

    for ticker, poids in derniere_selection.items():

        lignes.append({
            "Action": ACTIONS[ticker],
            "Ticker": ticker,
            "Poids (%)": round(float(poids) * 100, 2),
            "Dernier cours (€)": round(
                float(prix[ticker].dropna().iloc[-1]),
                2
            )
        })

    st.dataframe(
        pd.DataFrame(lignes),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Cette selection correspond a la derniere allocation "
        "calculee avec les informations de la veille."
    )

    st.divider()

    st.header("6. Parametres du backtest")

    st.write(f"Capital initial : {CAPITAL_INITIAL:.2f} €")
    st.write(f"Univers : {len(ACTIONS)} actions")
    st.write(f"Nombre de positions : {NB_ACTIONS}")
    st.write(f"Commission par transaction : {FRAIS * 100:.2f} %")
    st.write("Momentum court : 20 seances")
    st.write("Momentum long : 120 seances")
    st.write("Volatilite : 20 seances")
    st.write("Reequilibrage : quotidien")

    st.divider()

    st.warning(
        """
        LIMITES DU MODELE :

        - Les actions sont celles de la liste actuelle, ce qui
          introduit un biais de survivance.
        - Le CAC 40 utilise ici son indice de prix, sans
          reinvestissement des dividendes.
        - Les frais de 0,10 % ne comprennent pas les spreads,
          taxes eventuelles et glissements de prix.
        - Le modele autorise les fractions d'actions.
        - Les donnees historiques Yahoo Finance peuvent
          comporter des lacunes.
        - Le backtest quotidien ne reproduit pas les cinq
          decisions intraday du simulateur initial.
        """
    )

else:

    st.info(
        "Cliquez sur le bouton pour lancer votre premier "
        "backtest historique."
    )

st.caption(
    "Outil de simulation historique uniquement. "
    "Aucun ordre boursier reel n'est execute."
)
