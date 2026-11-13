
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Simulateur multi-actions – 500 € virtuels",
    layout="wide"
)

st.title("Simulateur multi-actions – capital virtuel de 500 €")
st.caption(
    "Simulation historique uniquement : aucune connexion à un courtier "
    "et aucun ordre réel."
)

# ---------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------

st.sidebar.header("Paramètres de simulation")

symbols_text = st.sidebar.text_area(
    "Actions à suivre (codes Yahoo Finance)",
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
    step=0.10,
    help="La vente n'est autorisée que si tous les frais sont couverts et que ce bénéfice minimum est atteint."
)

drop_pct = st.sidebar.number_input(
    "Recul du cours par rapport au plus haut récent (%)",
    min_value=0.1,
    value=1.5,
    step=0.1
)

lookback = st.sidebar.number_input(
    "Nombre de bougies pour calculer le plus haut",
    min_value=5,
    max_value=100,
    value=20
)

# Coût total estimé d'une transaction en pourcentage.
friction = (spread_pct + slippage_pct) / 100


# ---------------------------------------------------------
# TÉLÉCHARGEMENT DES DONNÉES
# ---------------------------------------------------------

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
    data = data.dropna(subset=["Open", "High", "Close"])

    return data


# ---------------------------------------------------------
# INDICATEURS ET SIGNAUX D'ACHAT
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------

def run_backtest(all_data):

    prepared = {}
    for symbol, data in all_data.items():
        if not data.empty:
            prepared[symbol] = add_indicators(data)

    if not prepared:
        return pd.DataFrame(), pd.DataFrame(), capital, [], prepared

    timeline = sorted(set().union(
        *[set(df.index) for df in prepared.values()]
    ))

    cash = float(capital)
    positions = []
    trades = []
    equity_curve = []

    for ts in timeline:

        # ---------------------------------------------
        # 1. EXAMINER LES POSITIONS DÉJÀ OUVERTES
        # ---------------------------------------------

        remaining_positions = []

        for pos in positions:

            symbol = pos["symbol"]
            df = prepared[symbol]

            if ts < pos["entry_time"]:
                remaining_positions.append(pos)
                continue

            if ts not in df.index:
                remaining_positions.append(pos)
                continue

            bar = df.loc[ts]

            # Prix de vente minimum permettant de couvrir :
            # coût total de l'achat + frais de vente + bénéfice cible.
            required_raw_exit = (
                pos["total_entry_cost"] + target_profit + fee
            ) / (pos["quantity"] * (1 - friction))

            # On vend uniquement si le plus haut de la bougie
            # atteint ce prix minimum.
            if float(bar["High"]) >= required_raw_exit:

                exit_price = required_raw_exit

                net_proceeds = (
                    pos["quantity"]
                    * exit_price
                    * (1 - friction)
                    - fee
                )

                net_profit = net_proceeds - pos["total_entry_cost"]

                # Sécurité supplémentaire : aucune vente déficitaire.
                if net_profit >= target_profit:

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

                    continue

            remaining_positions.append(pos)

        positions = remaining_positions

        # ---------------------------------------------
        # 2. RECHERCHER DE NOUVEAUX ACHATS
        # ---------------------------------------------

        for symbol, df in prepared.items():

            if len(positions) >= max_positions:
                break

            if ts not in df.index:
                continue

            # Pas de deuxième position sur la même action.
            if any(p["symbol"] == symbol for p in positions):
                continue

            loc = df.index.get_loc(ts)

            # Achat à l'ouverture de la bougie suivante.
            if loc + 1 >= len(df):
                continue

            row = df.iloc[loc]

            if not bool(row["BuySignal"]):
                continue

            next_row = df.iloc[loc + 1]
            entry_time = df.index[loc + 1]

            raw_entry_price = float(next_row["Open"])

            if not np.isfinite(raw_entry_price) or raw_entry_price <= 0:
                continue

            available_slots = max_positions - len(positions)

            # Répartition dynamique du capital disponible.
            slot_budget = cash / available_slots

            # Coût d'une unité, frais et friction d'achat compris.
            unit_cost = raw_entry_price * (1 + friction)

            quantity = (slot_budget - fee) / unit_cost

            if quantity <= 0:
                continue

            total_entry_cost = (
                quantity * unit_cost + fee
            )

            if total_entry_cost > cash:
                continue

            cash -= total_entry_cost

            positions.append({
                "symbol": symbol,
                "entry_time": entry_time,
                "entry_price": raw_entry_price,
                "quantity": quantity,
                "total_entry_cost": total_entry_cost
            })

        # ---------------------------------------------
        # 3. VALORISER LE PORTEFEUILLE
        # ---------------------------------------------

        market_value = 0.0

        for pos in positions:
            if ts < pos["entry_time"]:
                continue

            df = prepared[pos["symbol"]]
            available_data = df.loc[df.index <= ts]

            if not available_data.empty:
                last_price = float(available_data["Close"].iloc[-1])
                market_value += pos["quantity"] * last_price

        equity_curve.append({
            "Date": ts,
            "Liquidités (€)": cash,
            "Positions au cours du marché (€)": market_value,
            "Valeur portefeuille (€)": cash + market_value
        })

    # -----------------------------------------------------
    # 4. POSITIONS ENCORE OUVERTES À LA FIN DE L'HISTORIQUE
    # -----------------------------------------------------

    open_positions = []

    for pos in positions:

        df = prepared[pos["symbol"]]
        last_price = float(df["Close"].iloc[-1])

        gross_value = pos["quantity"] * last_price

        # Estimation de ce que rapporterait une liquidation
        # immédiate, frais de vente compris.
        estimated_net_sale = gross_value * (1 - friction) - fee

        unrealized_profit = (
            estimated_net_sale - pos["total_entry_cost"]
        )

        open_positions.append({
            "Action": pos["symbol"],
            "Date achat": pos["entry_time"],
            "Prix achat": pos["entry_price"],
            "Dernier cours": last_price,
            "Quantité": pos["quantity"],
            "Coût total achat (€)": pos["total_entry_cost"],
            "Valeur brute (€)": gross_value,
            "Valeur nette estimée (€)": estimated_net_sale,
            "Gain/perte latent estimé (€)": unrealized_profit
        })

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(equity_curve)
    open_df = pd.DataFrame(open_positions)

    return trades_df, curve_df, cash, open_df, prepared


# ---------------------------------------------------------
# LANCEMENT
# ---------------------------------------------------------

if st.button("Lancer la simulation", type="primary"):

    if not symbols:
        st.error("Indique au moins une action.")
        st.stop()

    with st.spinner("Téléchargement des cours et simulation..."):

        all_data = {}

        for symbol in symbols:
            try:
                data = download_one(symbol, period, interval)

                if not data.empty:
                    all_data[symbol] = data

            except Exception:
                pass

        if not all_data:
            st.error(
                "Aucune donnée téléchargée. Vérifie les symboles "
                "et les paramètres."
            )
            st.stop()

        trades_df, curve_df, cash, open_df, prepared = run_backtest(all_data)

    # -----------------------------------------------------
    # RÉSULTATS
    # -----------------------------------------------------

    st.subheader("Résultats de la simulation")

    realized_profit = 0.0

    if not trades_df.empty:
        realized_profit = trades_df["Bénéfice net (€)"].sum()

    unrealized_profit = 0.0

    if not open_df.empty:
        unrealized_profit = open_df[
            "Gain/perte latent estimé (€)"
        ].sum()

    open_net_value = 0.0

    if not open_df.empty:
        open_net_value = open_df[
            "Valeur nette estimée (€)"
        ].sum()

    estimated_final_value = cash + open_net_value

    total_result = estimated_final_value - capital

    winners = len(trades_df) if not trades_df.empty else 0

    win_rate = 100.0 if winners > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Capital initial",
        f"{capital:,.2f} €"
    )

    col2.metric(
        "Liquidités disponibles",
        f"{cash:,.2f} €"
    )

    col3.metric(
        "Bénéfices réalisés",
        f"{realized_profit:,.2f} €"
    )

    col4.metric(
        "Résultat estimé du portefeuille",
        f"{total_result:,.2f} €",
        delta=f"{total_result:,.2f} €"
    )

    st.caption(
        "Le résultat estimé tient compte de la valeur nette théorique "
        "des positions encore ouvertes, comme si elles étaient liquidées "
        "au dernier cours disponible."
    )

    st.divider()

    # -----------------------------------------------------
    # COURBE DU PORTEFEUILLE
    # -----------------------------------------------------

    st.subheader("Évolution du portefeuille")

    if not curve_df.empty:
        chart_data = curve_df.set_index("Date")[
            ["Valeur portefeuille (€)"]
        ]
        st.line_chart(chart_data)

    # -----------------------------------------------------
    # OPÉRATIONS RÉALISÉES
    # -----------------------------------------------------

    st.subheader("Ventes effectivement réalisées")

    if not trades_df.empty:

        st.dataframe(
            trades_df,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"{len(trades_df)} vente(s) réalisée(s). "
            f"Toutes les ventes respectent le bénéfice net minimum "
            f"de {target_profit:.2f} €."
        )

        csv_trades = trades_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Télécharger les opérations (CSV)",
            data=csv_trades,
            file_name="operations_simulation.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "Aucune vente n'a encore atteint le bénéfice net minimum."
        )

    # -----------------------------------------------------
    # POSITIONS ENCORE OUVERTES
    # -----------------------------------------------------

    st.subheader("Positions encore ouvertes")

    if not open_df.empty:

        st.dataframe(
            open_df,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "Ces positions ne sont pas vendues automatiquement. "
            "Le gain ou la perte affiché est une estimation au dernier cours."
        )

    else:
        st.info("Aucune position n'est encore ouverte.")

    # -----------------------------------------------------
    # RÉCAPITULATIF
    # -----------------------------------------------------

    st.subheader("Bilan")

    st.write(f"Capital initial : **{capital:.2f} €**")
    st.write(f"Liquidités restantes : **{cash:.2f} €**")
    st.write(f"Bénéfices réalisés : **{realized_profit:.2f} €**")
    st.write(f"Résultat latent estimé : **{unrealized_profit:.2f} €**")
    st.write(
        f"Valeur nette estimée du portefeuille : "
        f"**{estimated_final_value:.2f} €**"
    )

    st.caption(
        "Attention : une position peut rester ouverte très longtemps, "
        "voire jusqu'à la fin des données disponibles. Le capital investi "
        "dans cette position reste alors immobilisé."
    )

else:
    st.info(
        "Choisis tes paramètres dans la colonne de gauche, "
        "puis clique sur « Lancer la simulation »."
    )
