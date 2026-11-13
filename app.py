
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Simulateur multi-actions – 500 € virtuels",
    layout="wide"
)

st.title("Simulateur multi-actions – capital virtuel")
st.caption(
    "Simulation historique uniquement. Aucun ordre réel n'est envoyé."
)

# =========================================================
# 1. PARAMÈTRES
# =========================================================

st.sidebar.header("Paramètres")

symbols_text = st.sidebar.text_area(
    "Actions à suivre",
    "AIR.PA, OR.PA, SAN.PA, TTE.PA, SU.PA, AI.PA, BNP.PA, DG.PA, RI.PA, ACA.PA"
)

symbols = list(dict.fromkeys(
    s.strip().upper()
    for s in symbols_text.replace("\n", ",").split(",")
    if s.strip()
))

period = st.sidebar.selectbox(
    "Période historique",
    ["1mo", "3mo", "6mo", "1y"],
    index=1
)

interval = st.sidebar.selectbox(
    "Intervalle",
    ["15m", "30m", "60m", "1d"],
    index=0
)

capital = st.sidebar.number_input(
    "Capital virtuel (€)",
    min_value=100.0,
    max_value=100000.0,
    value=500.0,
    step=50.0
)

fee = st.sidebar.number_input(
    "Frais fixes par ordre (€)",
    min_value=0.0,
    value=1.0,
    step=0.10
)

spread_pct = st.sidebar.number_input(
    "Spread (%) par opération",
    min_value=0.0,
    value=0.05,
    step=0.01
)

slippage_pct = st.sidebar.number_input(
    "Slippage (%) par opération",
    min_value=0.0,
    value=0.02,
    step=0.01
)

max_positions = st.sidebar.slider(
    "Nombre maximum de positions simultanées",
    min_value=1,
    max_value=10,
    value=2
)

target_profit = st.sidebar.number_input(
    "Bénéfice net minimum par vente (€)",
    min_value=0.01,
    value=0.01,
    step=0.10
)

lookback = st.sidebar.number_input(
    "Nombre de bougies pour calculer le plus haut",
    min_value=5,
    max_value=100,
    value=20
)

drop_pct = st.sidebar.number_input(
    "Recul du cours par rapport au plus haut récent (%)",
    min_value=0.1,
    value=1.5,
    step=0.1
)

# Coût proportionnel modélisé à l'achat et à la vente.
friction = (spread_pct + slippage_pct) / 100.0

# Tolérance technique pour les comparaisons de nombres décimaux.
EPSILON = 1e-8


# =========================================================
# 2. TÉLÉCHARGEMENT DES COURS
# =========================================================

@st.cache_data(ttl=300)
def download_one(symbol, period, interval):

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    data = data[required].copy()

    data = data.dropna(
        subset=["Open", "High", "Close"]
    )

    # Élimination des dates en double et tri chronologique.
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    return data


# =========================================================
# 3. INDICATEURS ET SIGNAUX D'ACHAT
# =========================================================

def add_indicators(data):

    data = data.copy()

    data["PriorHigh"] = (
        data["High"]
        .rolling(int(lookback))
        .max()
        .shift(1)
    )

    delta = data["Close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))

    data["BuySignal"] = (
        (data["Close"] < data["PriorHigh"] * (1 - drop_pct / 100))
        & (data["Close"] > data["Close"].shift(1))
        & (data["RSI"] > data["RSI"].shift(1))
        & (data["RSI"] < 55)
    )

    return data


# =========================================================
# 4. CALCUL DU PRIX MINIMUM DE VENTE
# =========================================================

def minimum_sale_price(position):

    quantity = position["quantity"]
    total_cost = position["total_entry_cost"]

    # Il faut couvrir :
    # - le coût total de l'achat ;
    # - les frais fixes de vente ;
    # - le bénéfice net demandé.
    #
    # Une petite marge technique évite les erreurs d'arrondi.

    required = (
        total_cost + fee + target_profit + 0.000001
    ) / (quantity * (1 - friction))

    return required


# =========================================================
# 5. MOTEUR DE SIMULATION
# =========================================================

def run_backtest(all_data):

    prepared = {}

    for symbol, data in all_data.items():

        if not data.empty:
            prepared[symbol] = add_indicators(data)

    if not prepared:
        return None

    # Toutes les dates disponibles, tous symboles confondus.
    timeline = sorted(set().union(
        *[set(df.index) for df in prepared.values()]
    ))

    cash = float(capital)

    positions = []
    trades = []
    equity_curve = []

    # Dernier cours connu de chaque action.
    last_prices = {}

    # Dernier plus haut connu de chaque action.
    last_highs = {}

    # -----------------------------------------------------
    # Parcours chronologique
    # -----------------------------------------------------

    for ts in timeline:

        current_bars = {}

        # Récupération des bougies disponibles à cette date.
        for symbol, df in prepared.items():

            if ts in df.index:

                bar = df.loc[ts]

                current_bars[symbol] = bar

                last_prices[symbol] = float(bar["Close"])
                last_highs[symbol] = float(bar["High"])

        # -------------------------------------------------
        # A. EXAMEN DES POSITIONS OUVERTES
        # -------------------------------------------------

        remaining_positions = []

        for pos in positions:

            symbol = pos["symbol"]

            # Une action peut ne pas avoir de bougie à cette date.
            if symbol not in current_bars:
                remaining_positions.append(pos)
                continue

            # Ne jamais examiner une bougie antérieure à l'achat.
            if ts < pos["entry_time"]:
                remaining_positions.append(pos)
                continue

            bar = current_bars[symbol]

            high_price = float(bar["High"])

            sale_target = minimum_sale_price(pos)

            # Vérification du seuil de vente.
            # Le plus haut de la bougie doit atteindre le prix cible.
            if high_price + EPSILON >= sale_target:

                # Hypothèse de simulation :
                # vente au prix cible dès qu'il est touché.
                exit_price = sale_target

                net_proceeds = (
                    pos["quantity"]
                    * exit_price
                    * (1 - friction)
                    - fee
                )

                net_profit = (
                    net_proceeds - pos["total_entry_cost"]
                )

                # Deuxième contrôle indépendant :
                # on interdit toute vente déficitaire.
                if net_profit + EPSILON >= target_profit:

                    cash += net_proceeds

                    trades.append({
                        "Action": symbol,
                        "Date achat": pos["entry_time"],
                        "Date vente": ts,
                        "Prix achat": pos["entry_price"],
                        "Prix vente": exit_price,
                        "Quantité": pos["quantity"],
                        "Coût achat total": pos["total_entry_cost"],
                        "Produit net vente": net_proceeds,
                        "Bénéfice net (€)": net_profit,
                        "Résultat": "Gagnant"
                    })

                    # La position est vendue :
                    # elle ne figure plus dans les positions ouvertes.
                    continue

            # Si le seuil n'est pas atteint, on conserve la position.
            remaining_positions.append(pos)

        positions = remaining_positions

        # -------------------------------------------------
        # B. RECHERCHE DE NOUVEAUX ACHATS
        # -------------------------------------------------

        for symbol, df in prepared.items():

            if len(positions) >= max_positions:
                break

            if symbol not in current_bars:
                continue

            # Pas de nouvel achat si une position existe déjà.
            if any(p["symbol"] == symbol for p in positions):
                continue

            loc = df.index.get_loc(ts)

            # Achat à l'ouverture de la bougie suivante.
            if not isinstance(loc, (int, np.integer)):
                continue

            if loc + 1 >= len(df):
                continue

            row = current_bars[symbol]

            if not bool(row["BuySignal"]):
                continue

            next_row = df.iloc[loc + 1]

            entry_time = df.index[loc + 1]
            entry_price = float(next_row["Open"])

            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            available_slots = max_positions - len(positions)

            # Répartition du capital disponible.
            slot_budget = cash / available_slots

            # Coût unitaire réel modélisé.
            unit_cost = entry_price * (1 + friction)

            quantity = (slot_budget - fee) / unit_cost

            if quantity <= 0:
                continue

            total_entry_cost = quantity * unit_cost + fee

            if total_entry_cost > cash + EPSILON:
                continue

            cash -= total_entry_cost

            positions.append({
                "symbol": symbol,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "quantity": quantity,
                "total_entry_cost": total_entry_cost
            })

        # -------------------------------------------------
        # C. VALORISATION DU PORTEFEUILLE
        # -------------------------------------------------

        market_value = 0.0

        for pos in positions:

            symbol = pos["symbol"]

            if ts < pos["entry_time"]:
                continue

            if symbol in last_prices:

                market_value += (
                    pos["quantity"] * last_prices[symbol]
                )

        equity_curve.append({
            "Date": ts,
            "Liquidités (€)": cash,
            "Positions au cours du marché (€)": market_value,
            "Valeur portefeuille (€)": cash + market_value
        })

    # =====================================================
    # 6. POSITIONS RESTANT OUVERTES
    # =====================================================

    open_positions = []

    for pos in positions:

        symbol = pos["symbol"]

        if symbol not in last_prices:
            continue

        last_price = last_prices[symbol]
        last_high = last_highs[symbol]

        gross_value = pos["quantity"] * last_price

        estimated_net_sale = (
            gross_value * (1 - friction) - fee
        )

        unrealized_profit = (
            estimated_net_sale - pos["total_entry_cost"]
        )

        sale_target = minimum_sale_price(pos)

        # Ce contrôle permet de voir si le seuil était atteint
        # sur la dernière bougie disponible.
        last_bar_reached_target = (
            last_high + EPSILON >= sale_target
        )

        open_positions.append({

            "Action": symbol,

            "Date achat": pos["entry_time"],

            "Prix achat": pos["entry_price"],

            "Dernier cours": last_price,

            "Dernier plus haut": last_high,

            "Quantité": pos["quantity"],

            "Coût total achat": pos["total_entry_cost"],

            "Prix minimum de vente": sale_target,

            "Valeur brute": gross_value,

            "Valeur nette estimée": estimated_net_sale,

            "Gain/perte latent estimé (€)": unrealized_profit,

            "Seuil atteint dernière bougie": (
                "OUI" if last_bar_reached_target else "NON"
            )
        })

    trades_df = pd.DataFrame(trades)

    curve_df = pd.DataFrame(equity_curve)

    open_df = pd.DataFrame(open_positions)

    return {
        "trades": trades_df,
        "curve": curve_df,
        "cash": cash,
        "open": open_df,
        "prepared": prepared
    }


# =========================================================
# 7. INTERFACE ET RÉSULTATS
# =========================================================

if st.button("Lancer la simulation", type="primary"):

    if not symbols:
        st.error("Indique au moins une action.")
        st.stop()

    all_data = {}

    with st.spinner("Téléchargement des cours et simulation..."):

        for symbol in symbols:

            try:

                data = download_one(
                    symbol,
                    period,
                    interval
                )

                if not data.empty:
                    all_data[symbol] = data

            except Exception as error:

                st.warning(
                    f"Impossible de télécharger {symbol} : {error}"
                )

        if not all_data:
            st.error("Aucune donnée disponible.")
            st.stop()

        result = run_backtest(all_data)

    if result is None:
        st.error("La simulation n'a pas pu être réalisée.")
        st.stop()

    trades_df = result["trades"]
    curve_df = result["curve"]
    cash = result["cash"]
    open_df = result["open"]

    # -----------------------------------------------------
    # CALCUL DES RÉSULTATS
    # -----------------------------------------------------

    realized_profit = 0.0

    if not trades_df.empty:
        realized_profit = trades_df["Bénéfice net (€)"].sum()

    unrealized_profit = 0.0
    open_net_value = 0.0

    if not open_df.empty:

        unrealized_profit = (
            open_df["Gain/perte latent estimé (€)"].sum()
        )

        open_net_value = (
            open_df["Valeur nette estimée"].sum()
        )

    estimated_final_value = cash + open_net_value

    total_result = estimated_final_value - capital

    # -----------------------------------------------------
    # AFFICHAGE DES INDICATEURS
    # -----------------------------------------------------

    st.subheader("Résultats de la simulation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Capital initial",
        f"{capital:.2f} €"
    )

    col2.metric(
        "Liquidités disponibles",
        f"{cash:.2f} €"
    )

    col3.metric(
        "Bénéfices réalisés",
        f"{realized_profit:.2f} €"
    )

    col4.metric(
        "Résultat total estimé",
        f"{total_result:+.2f} €"
    )

    st.write(
        f"Nombre de ventes réalisées : **{len(trades_df)}**"
    )

    st.write(
        f"Nombre de positions encore ouvertes : **{len(open_df)}**"
    )

    st.divider()

    # -----------------------------------------------------
    # COURBE DU PORTEFEUILLE
    # -----------------------------------------------------

    st.subheader("Évolution du portefeuille")

    if not curve_df.empty:

        st.line_chart(
            curve_df.set_index("Date")[
                ["Valeur portefeuille (€)"]
            ]
        )

    # -----------------------------------------------------
    # VENTES RÉALISÉES
    # -----------------------------------------------------

    st.subheader("Ventes réalisées")

    if not trades_df.empty:

        st.dataframe(
            trades_df,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"{len(trades_df)} vente(s) réalisée(s)."
        )

        st.download_button(
            "Télécharger les ventes en CSV",
            data=trades_df.to_csv(index=False).encode("utf-8"),
            file_name="ventes_simulation.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Aucune vente n'a encore atteint le bénéfice minimum."
        )

    # -----------------------------------------------------
    # POSITIONS OUVERTES
    # -----------------------------------------------------

    st.subheader("Positions encore ouvertes")

    if not open_df.empty:

        st.dataframe(
            open_df,
            use_container_width=True,
            hide_index=True
        )

        # Alerte si le seuil est déjà atteint sur la dernière bougie.
        reached = open_df[
            open_df["Seuil atteint dernière bougie"] == "OUI"
        ]

        if not reached.empty:

            st.error(
                "ANOMALIE : une position ouverte a atteint son seuil "
                "de vente sur sa dernière bougie. Il faut examiner "
                "la simulation."
            )

        else:

            st.info(
                "Les seuils de vente n'ont pas été atteints sur "
                "la dernière bougie disponible pour ces positions."
            )

    else:

        st.success("Toutes les positions ont été vendues.")

    # -----------------------------------------------------
    # BILAN FINAL
    # -----------------------------------------------------

    st.divider()

    st.subheader("Bilan financier")

    st.write(f"Capital de départ : **{capital:.2f} €**")

    st.write(f"Liquidités disponibles : **{cash:.2f} €**")

    st.write(f"Bénéfices réalisés : **{realized_profit:.2f} €**")

    st.write(
        f"Résultat latent estimé : **{unrealized_profit:+.2f} €**"
    )

    st.write(
        f"Valeur nette estimée du portefeuille : "
        f"**{estimated_final_value:.2f} €**"
    )

    st.write(
        f"Résultat total estimé : **{total_result:+.2f} €**"
    )

    st.caption(
        "Les positions ouvertes ne sont pas liquidées artificiellement "
        "à la fin de la période. Leur valeur nette est estimée après "
        "frais de vente. Les ventes simulées supposent qu'un ordre "
        "limite est exécuté lorsque le plus haut d'une bougie atteint "
        "le prix cible ; cela ne garantit pas une exécution réelle."
    )

else:

    st.info(
        "Choisis tes paramètres puis clique sur "
        "« Lancer la simulation »."
    )
