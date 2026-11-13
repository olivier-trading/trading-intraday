import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# SIMULATEUR UNIQUEMENT :
# Ce programme télécharge des cours historiques et calcule des opérations fictives.
# Il ne contient aucune connexion à un courtier et aucune fonction d'envoi d'ordre.

st.set_page_config(page_title="Simulateur multi-actions – 500 € virtuels", layout="wide")
st.title("Simulateur multi-actions – capital virtuel de 500 €")
st.caption(
    "BACKTEST UNIQUEMENT — cours historiques Yahoo Finance, portefeuille fictif. "
    "Aucun compte bancaire ou courtier n'est connecté et aucun ordre réel n'est envoyé."
)

with st.sidebar:
    st.header("Paramètres de simulation")
    symbols_text = st.text_area(
        "Actions à surveiller (tickers Yahoo Finance, séparés par des virgules)",
        value="AIR.PA, OR.PA, SAN.PA, TTE.PA, SU.PA, AI.PA, BNP.PA, DG.PA, RI.PA, ACA.PA",
        height=110,
    )
    period = st.selectbox("Historique", ["1mo", "3mo", "6mo"], index=1)
    interval = st.selectbox("Intervalle", ["15m", "30m", "60m", "1d"], index=0)

    st.subheader("Capital et frais")
    capital = st.number_input("Capital initial virtuel (€)", min_value=500.0, max_value=500.0, value=500.0, step=10.0)
    fee = st.number_input("Frais fixes par achat ou vente (€)", min_value=0.0, value=1.0, step=0.5)
    spread_pct = st.number_input("Spread estimé par côté (%)", min_value=0.0, value=0.05, step=0.01)
    slippage_pct = st.number_input("Glissement estimé par côté (%)", min_value=0.0, value=0.02, step=0.01)

    st.subheader("Règles de portefeuille")
    max_positions = st.slider("Nombre maximal de positions simultanées", 1, 5, 2)
    target_profit = st.number_input("Bénéfice net minimum visé par opération (€)", min_value=0.10, value=1.0, step=0.25)
    stop_pct = st.number_input("Stop de sécurité (%)", min_value=0.5, max_value=15.0, value=4.0, step=0.5)
    max_holding_bars = st.number_input("Durée maximale d'une position (nombre de bougies)", min_value=2, max_value=500, value=26, step=1)

    st.subheader("Détection des baisses")
    lookback = st.number_input("Fenêtre de recherche du plus haut (bougies)", min_value=5, max_value=100, value=20, step=1)
    drop_pct = st.number_input("Baisse minimale depuis ce plus haut (%)", min_value=0.5, max_value=20.0, value=1.5, step=0.25)

@st.cache_data(ttl=300, show_spinner=False)
def download_one(symbol, period, interval):
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).title() for c in data.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in needed):
        return pd.DataFrame()
    return data[needed].dropna().copy()

def add_indicators(data, lookback):
    d = data.copy()
    d["PriorHigh"] = d["High"].rolling(int(lookback)).max().shift(1)
    delta = d["Close"].diff()
    gains = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gains / losses.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    d["RSI_prev"] = d["RSI"].shift(1)
    d["Close_prev"] = d["Close"].shift(1)
    d = d.dropna().copy()
    # Baisse depuis un plus haut récent, suivie d'un début de rebond.
    d["BuySignal"] = (
        (d["Close"] <= d["PriorHigh"] * (1 - drop_pct / 100))
        & (d["Close"] > d["Close_prev"])
        & (d["RSI"] > d["RSI_prev"])
        & (d["RSI"] < 55)
    )
    return d

def run_backtest(raw_data, settings):
    """Backtest multi-actions avec exécution simulée à l'ouverture de la bougie suivante."""
    prepared = {}
    for symbol, frame in raw_data.items():
        if not frame.empty:
            prepared[symbol] = add_indicators(frame, settings["lookback"])
    if not prepared:
        raise ValueError("Aucune série exploitable. Vérifiez les tickers et la période.")

    # Événements chronologiques communs à toutes les actions.
    timeline = sorted(set().union(*(set(df.index) for df in prepared.values())))
    cash = float(settings["capital"])
    positions = {}
    trades = []
    equity_points = []
    friction = (settings["spread_pct"] + settings["slippage_pct"]) / 100
    slot_budget = settings["capital"] / settings["max_positions"]

    for ts in timeline:
        # 1) Gérer les positions déjà ouvertes.
        for symbol in list(positions.keys()):
            df = prepared[symbol]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            pos = positions[symbol]
            pos["bars_held"] += 1

            low = float(row["Low"])
            high = float(row["High"])
            close = float(row["Close"])
            exit_reason = None
            raw_exit = None

            # Si stop et objectif sont touchés dans la même bougie, hypothèse prudente : stop d'abord.
            if low <= pos["stop_price"]:
                exit_reason, raw_exit = "Stop de sécurité", pos["stop_price"]
            else:
                # Prix minimal permettant d'atteindre le bénéfice net demandé, après frais et friction.
                required_sale = (
                    pos["total_entry_cost"] + settings["target_profit"] + settings["fee"]
                ) / (pos["quantity"] * (1 - friction))
                if high >= required_sale:
                    exit_reason, raw_exit = "Objectif net atteint", required_sale
                elif pos["bars_held"] >= settings["max_holding_bars"]:
                    exit_reason, raw_exit = "Durée maximale", close

            if exit_reason is not None:
                effective_sale = raw_exit * (1 - friction)
                net_proceeds = pos["quantity"] * effective_sale - settings["fee"]
                pnl = net_proceeds - pos["total_entry_cost"]
                cash += net_proceeds
                trades.append({
                    "Action": symbol,
                    "Entrée": pos["entry_time"],
                    "Sortie": ts,
                    "Prix achat": pos["entry_price"],
                    "Prix vente": raw_exit,
                    "Quantité simulée": pos["quantity"],
                    "Motif": exit_reason,
                    "Résultat net (€)": pnl,
                    "Rendement net (%)": 100 * pnl / pos["total_entry_cost"],
                })
                del positions[symbol]

        # 2) Chercher de nouvelles entrées. Le signal est connu à la clôture de la bougie,
        # l'achat simulé se fait à l'ouverture de la bougie suivante.
        for symbol, df in prepared.items():
            if symbol in positions or len(positions) >= settings["max_positions"]:
                continue
            loc = df.index.get_indexer([ts])[0]
            if loc < 0 or loc + 1 >= len(df):
                continue
            row = df.iloc[loc]
            next_row = df.iloc[loc + 1]
            if not bool(row["BuySignal"]):
                continue

            available_slots = settings["max_positions"] - len(positions)
            budget = min(slot_budget, cash)
            if available_slots <= 0 or budget <= settings["fee"]:
                continue

            entry_price = float(next_row["Open"])
            effective_entry = entry_price * (1 + friction)
            quantity = (budget - settings["fee"]) / effective_entry
            total_cost = quantity * effective_entry + settings["fee"]
            if quantity <= 0 or total_cost > cash:
                continue

            cash -= total_cost
            positions[symbol] = {
                "entry_time": df.index[loc + 1],
                "entry_price": entry_price,
                "quantity": quantity,
                "total_entry_cost": total_cost,
                "stop_price": entry_price * (1 - settings["stop_pct"] / 100),
                "bars_held": 0,
            }

        # 3) Valorisation indicative au dernier cours connu de chaque position.
        market_value = 0.0
        for symbol, pos in positions.items():
            df = prepared[symbol]
            eligible = df.loc[df.index <= ts]
            if not eligible.empty:
                last_price = float(eligible.iloc[-1]["Close"])
                market_value += pos["quantity"] * last_price
        equity_points.append((ts, cash + market_value))

    # Fermer fictivement les positions restantes à la dernière clôture disponible.
    for symbol, pos in list(positions.items()):
        df = prepared[symbol]
        last = df.iloc[-1]
        ts = df.index[-1]
        raw_exit = float(last["Close"])
        net_proceeds = pos["quantity"] * raw_exit * (1 - friction) - settings["fee"]
        pnl = net_proceeds - pos["total_entry_cost"]
        cash += net_proceeds
        trades.append({
            "Action": symbol,
            "Entrée": pos["entry_time"],
            "Sortie": ts,
            "Prix achat": pos["entry_price"],
            "Prix vente": raw_exit,
            "Quantité simulée": pos["quantity"],
            "Motif": "Fin de l'historique",
            "Résultat net (€)": pnl,
            "Rendement net (%)": 100 * pnl / pos["total_entry_cost"],
        })

    trades_df = pd.DataFrame(trades)
    curve = pd.DataFrame(equity_points, columns=["Date", "Capital estimé"]).set_index("Date")
    return trades_df, curve, cash, prepared

if st.button("Lancer le backtest multi-actions", type="primary"):
    symbols = list(dict.fromkeys(
        s.strip().upper() for s in symbols_text.replace("\n", ",").split(",") if s.strip()
    ))
    if not symbols:
        st.error("Ajoutez au moins un ticker.")
        st.stop()

    settings = {
        "capital": capital,
        "fee": fee,
        "spread_pct": spread_pct,
        "slippage_pct": slippage_pct,
        "max_positions": max_positions,
        "target_profit": target_profit,
        "stop_pct": stop_pct,
        "max_holding_bars": int(max_holding_bars),
        "lookback": int(lookback),
    }
    raw_data = {}
    failed = []
    with st.spinner("Téléchargement des cours historiques et simulation en cours..."):
        for symbol in symbols:
            try:
                frame = download_one(symbol, period, interval)
                if frame.empty or len(frame) < int(lookback) + 5:
                    failed.append(symbol)
                else:
                    raw_data[symbol] = frame
            except Exception:
                failed.append(symbol)

    if failed:
        st.warning("Données indisponibles ou insuffisantes pour : " + ", ".join(failed))
    if not raw_data:
        st.error("Aucune donnée récupérée. Vérifiez les tickers ou choisissez une autre période.")
        st.stop()

    try:
        trades, curve, final_value, prepared = run_backtest(raw_data, settings)
        pnl = final_value - capital
        ret = 100 * pnl / capital
        wins = int((trades["Résultat net (€)"] > 0).sum()) if not trades.empty else 0
        losses = int((trades["Résultat net (€)"] <= 0).sum()) if not trades.empty else 0
        win_rate = 100 * wins / len(trades) if len(trades) else 0

        st.subheader("Résultats de la simulation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital final", f"{final_value:,.2f} €".replace(",", " "))
        c2.metric("Résultat net", f"{pnl:+,.2f} €".replace(",", " "), f"{ret:+.2f} %")
        c3.metric("Opérations clôturées", len(trades))
        c4.metric("Opérations gagnantes", f"{win_rate:.1f} %", f"{wins} gagnantes / {losses} perdantes")

        st.line_chart(curve["Capital estimé"])
        st.caption(
            f"{len(prepared)} actions effectivement testées. "
            f"Positions simultanées maximales : {max_positions}. "
            "Les quantités fractionnaires sont autorisées dans le portefeuille fictif."
        )

        if not trades.empty:
            st.subheader("Historique des opérations")
            st.dataframe(trades.sort_values("Entrée"), use_container_width=True)
            st.download_button(
                "Télécharger les opérations (CSV)",
                trades.to_csv(index=False).encode("utf-8"),
                "backtest_multi_actions.csv",
                "text/csv",
            )
        else:
            st.info("Aucun signal d'achat n'a été détecté sur la période.")

        st.subheader("Limites à connaître")
        st.warning(
            "Un backtest ne prédit pas les performances futures. Les cours historiques, "
            "le spread et le glissement sont approximatifs. Les exécutions sont simulées "
            "à partir des bougies Yahoo Finance. Ce programme ne passe aucun ordre réel."
        )
    except Exception as exc:
        st.error(f"Erreur pendant le backtest : {exc}")

st.divider()
st.caption(
    "SIMULATION SEULEMENT. Le programme ne contient aucun module de connexion à un courtier, "
    "aucune clé API de trading et aucune fonction d'envoi d'ordres."
)
