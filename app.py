
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Simulateur de sélection et trading",
    layout="wide"
)

st.title("Simulateur multi-actions – Sélection et réinvestissement")
st.caption(
    "Simulation historique uniquement. Aucun ordre réel n'est envoyé."
)

# ============================================================
# PARAMÈTRES UTILISATEUR
# ============================================================

st.sidebar.header("Paramètres de simulation")

capital_initial = st.sidebar.number_input(
    "Capital initial (€)",
    min_value=100.0,
    max_value=100000.0,
    value=500.0,
    step=50.0
)

frais = st.sidebar.number_input(
    "Frais fixes par ordre (€)",
    min_value=0.0,
    value=1.0,
    step=0.25
)

spread_pct = st.sidebar.number_input(
    "Spread estimé (%)",
    min_value=0.0,
    value=0.05,
    step=0.01
)

slippage_pct = st.sidebar.number_input(
    "Glissement estimé (%)",
    min_value=0.0,
    value=0.02,
    step=0.01
)

benefice_minimum = st.sidebar.number_input(
    "Bénéfice net minimum pour vendre (€)",
    min_value=0.01,
    value=0.50,
    step=0.10
)

max_positions = st.sidebar.number_input(
    "Nombre maximum de positions simultanées",
    min_value=1,
    max_value=10,
    value=2
)

score_minimum = st.sidebar.slider(
    "Score minimum pour acheter",
    min_value=40,
    max_value=90,
    value=65
)

controles_journaliers = st.sidebar.selectbox(
    "Contrôles par journée de Bourse",
    [3, 4, 5, 6],
    index=2
)

periode = st.sidebar.selectbox(
    "Période historique",
    ["30d", "60d"],
    index=1
)

# ============================================================
# ACTIONS ANALYSÉES
# ============================================================

actions = {
    "AIR.PA": "Airbus",
    "OR.PA": "L'Oréal",
    "SAN.PA": "Sanofi",
    "TTE.PA": "TotalEnergies",
    "SU.PA": "Schneider Electric",
    "AI.PA": "Air Liquide",
    "BNP.PA": "BNP Paribas",
    "DG.PA": "Vinci",
    "RI.PA": "Pernod Ricard",
    "ACA.PA": "Crédit Agricole"
}

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================

def calculer_rsi(prix, periode=14):
    delta = prix.diff()

    gains = delta.clip(lower=0)
    pertes = -delta.clip(upper=0)

    moyenne_gains = gains.rolling(periode).mean()
    moyenne_pertes = pertes.rolling(periode).mean()

    rs = moyenne_gains / moyenne_pertes.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calculer_indicateurs(df):
    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    df["Momentum10"] = (
        df["Close"] / df["Close"].shift(10) - 1
    ) * 100

    df["RSI"] = calculer_rsi(df["Close"])

    df["VolumeMoyen"] = df["Volume"].rolling(20).mean()

    df["VolumeRatio"] = (
        df["Volume"] / df["VolumeMoyen"].replace(0, np.nan)
    )

    return df


# ============================================================
# CALCUL DU SCORE DE SÉLECTION
# ============================================================

def calculer_score(ligne):
    score = 0

    prix = ligne["Close"]
    sma20 = ligne["SMA20"]
    sma50 = ligne["SMA50"]
    momentum = ligne["Momentum10"]
    rsi = ligne["RSI"]
    volume_ratio = ligne["VolumeRatio"]

    if pd.isna(sma20) or pd.isna(sma50):
        return 0

    # Tendance : maximum 30 points
    if prix > sma20:
        score += 15

    if sma20 > sma50:
        score += 15

    # Dynamique récente : maximum 25 points
    if pd.notna(momentum):
        if momentum > 0:
            score += min(25, momentum * 5)

    # RSI : maximum 25 points
    if pd.notna(rsi):
        if 50 <= rsi <= 65:
            score += 25
        elif 45 <= rsi < 50 or 65 < rsi <= 70:
            score += 15
        elif 40 <= rsi < 45 or 70 < rsi <= 75:
            score += 5

    # Volume : maximum 20 points
    if pd.notna(volume_ratio):
        if volume_ratio >= 1.5:
            score += 20
        elif volume_ratio >= 1.1:
            score += 15
        elif volume_ratio >= 0.8:
            score += 10

    return min(100, round(score, 1))


# ============================================================
# TÉLÉCHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data(ttl=900)
def telecharger_donnees(symboles, periode):
    donnees = {}

    progression = st.progress(0)
    statut = st.empty()

    for i, symbole in enumerate(symboles):
        statut.write(f"Téléchargement de {symbole}...")

        try:
            df = yf.download(
                symbole,
                period=periode,
                interval="1h",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if df is None or df.empty:
                continue

            # Gestion des colonnes MultiIndex de Yahoo Finance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            colonnes = ["Open", "High", "Low", "Close", "Volume"]

            if not all(c in df.columns for c in colonnes):
                continue

            df = df[colonnes].copy()
            df = df.dropna(subset=["Close"])

            if len(df) < 60:
                continue

            if df.index.tz is not None:
                df.index = df.index.tz_convert("Europe/Paris")

            df = calculer_indicateurs(df)

            df["Score"] = df.apply(calculer_score, axis=1)

            # Sélection de cinq points de contrôle par journée
            # parmi les bougies horaires disponibles.
            groupes = []

            for _, groupe in df.groupby(df.index.date):
                if len(groupe) <= 0:
                    continue

                nombre = min(
                    controles_journaliers,
                    len(groupe)
                )

                indices = np.linspace(
                    0,
                    len(groupe) - 1,
                    nombre
                ).round().astype(int)

                groupes.append(groupe.iloc[indices])

            if groupes:
                donnees[symbole] = pd.concat(groupes)

        except Exception as e:
            st.warning(f"Erreur pour {symbole} : {e}")

        progression.progress((i + 1) / len(symboles))

    statut.empty()
    progression.empty()

    return donnees


# ============================================================
# SIMULATION
# ============================================================

def lancer_simulation(donnees):

    friction = (spread_pct + slippage_pct) / 100

    cash = float(capital_initial)

    positions = {}
    historique_ventes = []
    historique_achats = []
    historique_portefeuille = []

    # Construction de la chronologie commune
    evenements = []

    for symbole, df in donnees.items():
        for date, ligne in df.iterrows():
            evenements.append({
                "date": date,
                "symbole": symbole,
                "ligne": ligne
            })

    evenements.sort(key=lambda x: x["date"])

    if not evenements:
        return None

    dates = sorted(set(e["date"] for e in evenements))

    # Traitement des contrôles chronologiquement
    for date in dates:

        observations = [
            e for e in evenements
            if e["date"] == date
        ]

        cours_disponibles = {}

        for observation in observations:
            symbole = observation["symbole"]
            ligne = observation["ligne"]

            cours = float(ligne["Close"])
            score = float(ligne["Score"])

            cours_disponibles[symbole] = {
                "cours": cours,
                "score": score
            }

        # ----------------------------------------------------
        # 1. VENDRE LES POSITIONS RENTABLES
        # ----------------------------------------------------

        for symbole in list(positions.keys()):

            if symbole not in cours_disponibles:
                continue

            cours = cours_disponibles[symbole]["cours"]
            position = positions[symbole]

            quantite = position["quantite"]
            cout_achat_total = position["cout_total"]

            prix_vente = cours * (1 - friction)

            produit_vente = (
                quantite * prix_vente - frais
            )

            benefice_net = (
                produit_vente - cout_achat_total
            )

            if benefice_net >= benefice_minimum:

                cash += produit_vente

                historique_ventes.append({
                    "Date": date,
                    "Action": symbole,
                    "Nom": actions[symbole],
                    "Prix achat": position["prix_achat"],
                    "Prix vente": prix_vente,
                    "Quantité": quantite,
                    "Bénéfice net": benefice_net,
                    "Durée (heures)": (
                        date - position["date_achat"]
                    ).total_seconds() / 3600
                })

                del positions[symbole]

        # ----------------------------------------------------
        # 2. RECHERCHER LES MEILLEURES ACTIONS À ACHETER
        # ----------------------------------------------------

        places_disponibles = (
            max_positions - len(positions)
        )

        if places_disponibles > 0 and cash > frais:

            candidates = []

            for symbole, info in cours_disponibles.items():

                if symbole in positions:
                    continue

                score = info["score"]

                if score >= score_minimum:
                    candidates.append({
                        "symbole": symbole,
                        "score": score,
                        "cours": info["cours"]
                    })

            # Les meilleurs scores sont prioritaires
            candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            for candidat in candidates:

                if len(positions) >= max_positions:
                    break

                symbole = candidat["symbole"]
                cours = candidat["cours"]

                places_restantes = (
                    max_positions - len(positions)
                )

                budget = cash / places_restantes

                prix_achat = cours * (1 + friction)

                quantite = (
                    budget - frais
                ) / prix_achat

                if quantite <= 0:
                    continue

                cout_total = (
                    quantite * prix_achat + frais
                )

                if cout_total > cash:
                    continue

                cash -= cout_total

                positions[symbole] = {
                    "quantite": quantite,
                    "prix_achat": prix_achat,
                    "cout_total": cout_total,
                    "date_achat": date,
                    "score_achat": candidat["score"]
                }

                historique_achats.append({
                    "Date": date,
                    "Action": symbole,
                    "Nom": actions[symbole],
                    "Prix achat": prix_achat,
                    "Quantité": quantite,
                    "Coût total": cout_total,
                    "Score": candidat["score"]
                })

        # ----------------------------------------------------
        # 3. VALORISER LE PORTEFEUILLE
        # ----------------------------------------------------

        valeur_positions = 0

        for symbole, position in positions.items():

            if symbole in cours_disponibles:
                cours = cours_disponibles[symbole]["cours"]
            else:
                cours = position["prix_achat"]

            valeur_positions += (
                position["quantite"] * cours * (1 - friction)
            )

        valeur_totale = cash + valeur_positions

        historique_portefeuille.append({
            "Date": date,
            "Liquidités": cash,
            "Valeur positions": valeur_positions,
            "Valeur totale": valeur_totale
        })

    return {
        "cash": cash,
        "positions": positions,
        "ventes": historique_ventes,
        "achats": historique_achats,
        "portefeuille": historique_portefeuille,
        "donnees": donnees
    }


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

if st.button("LANCER LA SIMULATION", type="primary"):

    st.info("Téléchargement et analyse des actions en cours...")

    donnees = telecharger_donnees(
        list(actions.keys()),
        periode
    )

    if not donnees:
        st.error(
            "Aucune donnée exploitable n'a été téléchargée. "
            "Réessaie ultérieurement."
        )

    else:

        resultat = lancer_simulation(donnees)

        if resultat is None:
            st.error("Impossible de lancer la simulation.")

        else:

            ventes = pd.DataFrame(resultat["ventes"])
            achats = pd.DataFrame(resultat["achats"])
            portefeuille = pd.DataFrame(resultat["portefeuille"])

            cash = resultat["cash"]
            positions = resultat["positions"]

            valeur_positions = 0

            for symbole, position in positions.items():

                df = donnees[symbole]
                dernier_cours = float(df["Close"].iloc[-1])

                valeur_positions += (
                    position["quantite"]
                    * dernier_cours
                    * (1 - (spread_pct + slippage_pct) / 100)
                )

            valeur_finale = cash + valeur_positions

            benefice_realise = (
                ventes["Bénéfice net"].sum()
                if not ventes.empty else 0
            )

            benefice_total = (
                valeur_finale - capital_initial
            )

            st.header("Résultats de la simulation")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Capital initial",
                f"{capital_initial:.2f} €"
            )

            c2.metric(
                "Valeur finale estimée",
                f"{valeur_finale:.2f} €",
                f"{benefice_total:+.2f} €"
            )

            c3.metric(
                "Bénéfice réalisé",
                f"{benefice_realise:.2f} €"
            )

            c4.metric(
                "Nombre de ventes",
                len(ventes)
            )

            st.subheader("Évolution du portefeuille")

            if not portefeuille.empty:

                graphique = portefeuille.set_index("Date")[
                    ["Valeur totale"]
                ]

                st.line_chart(graphique)

            # ------------------------------------------------
            # CLASSEMENT DES ACTIONS À LA FIN
            # ------------------------------------------------

            st.subheader("Classement des actions")

            classement = []

            for symbole, df in donnees.items():

                derniere_ligne = df.iloc[-1]

                classement.append({
                    "Action": symbole,
                    "Nom": actions[symbole],
                    "Dernier cours": float(derniere_ligne["Close"]),
                    "Score": float(derniere_ligne["Score"]),
                    "RSI": float(derniere_ligne["RSI"]),
                    "Momentum (%)": float(
                        derniere_ligne["Momentum10"]
                    )
                })

            classement = pd.DataFrame(classement)

            classement = classement.sort_values(
                "Score",
                ascending=False
            )

            st.dataframe(
                classement.round(2),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # ACHATS
            # ------------------------------------------------

            st.subheader("Historique des achats")

            if not achats.empty:
                st.dataframe(
                    achats.round(3),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("Aucun achat réalisé.")

            # ------------------------------------------------
            # VENTES
            # ------------------------------------------------

            st.subheader("Historique des ventes")

            if not ventes.empty:

                st.dataframe(
                    ventes.round(3),
                    use_container_width=True,
                    hide_index=True
                )

                st.write(
                    f"Nombre de ventes : {len(ventes)}"
                )

                st.write(
                    f"Bénéfice moyen par vente : "
                    f"{ventes['Bénéfice net'].mean():.2f} €"
                )

                st.write(
                    f"Meilleure vente : "
                    f"{ventes['Bénéfice net'].max():.2f} €"
                )

                st.write(
                    f"Vente la moins bénéficiaire : "
                    f"{ventes['Bénéfice net'].min():.2f} €"
                )

            else:
                st.write("Aucune vente réalisée.")

            # ------------------------------------------------
            # POSITIONS ENCORE OUVERTES
            # ------------------------------------------------

            st.subheader("Positions encore ouvertes")

            positions_ouvertes = []

            for symbole, position in positions.items():

                dernier_cours = float(
                    donnees[symbole]["Close"].iloc[-1]
                )

                valeur_nette = (
                    position["quantite"]
                    * dernier_cours
                    * (1 - (spread_pct + slippage_pct) / 100)
                    - frais
                )

                gain_latent = (
                    valeur_nette - position["cout_total"]
                )

                positions_ouvertes.append({
                    "Action": symbole,
                    "Nom": actions[symbole],
                    "Prix achat": position["prix_achat"],
                    "Dernier cours": dernier_cours,
                    "Gain latent estimé": gain_latent,
                    "Score à l'achat": position["score_achat"]
                })

            if positions_ouvertes:

                st.dataframe(
                    pd.DataFrame(positions_ouvertes).round(3),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.write("Aucune position ouverte.")

            st.success("Simulation terminée.")

            st.caption(
                "Les résultats sont historiques et théoriques. "
                "Ils ne garantissent pas les performances futures. "
                "Les données horaires Yahoo Finance peuvent être "
                "incomplètes ou retardées."
            )

else:

    st.write(
        "Configure les paramètres dans la colonne de gauche, "
        "puis clique sur LANCER LA SIMULATION."
    )
