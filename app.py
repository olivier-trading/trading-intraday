
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Simulateur intraday – 1 00 €", layout="wide")
st.title("Simulateur de trading intraday capital de 100 euros")
st.caption("Mode simulation uniquement — aucun ordre réel n'est envoyé à Trade Republic.")

with st.sidebar:
    st.header("Paramètres")
    ticker = st.text_input("Symbole Yahoo Finance", "AIR.PA")
    period = st.selectbox("Historique", ["5d", "1mo", "3mo", "6mo"], index=1)
    interval = st.selectbox("Intervalle", ["5m", "15m", "30m", "60m"], index=0)
    capital = st.number_input("Capital initial (€)", min_value=100.0, value=100.0, step=100.0)
    fee = st.number_input("Frais fixes par ordre (€)", min_value=0.0, value=1.0, step=0.5)
    spread_pct = st.number_input("Spread estimé par côté (%)", min_value=0.0, value=0.05, step=0.01)
    slippage_pct = st.number_input("Glissement estimé par côté (%)", min_value=0.0, value=0.02, step=0.01)
    max_exposure = st.slider("Exposition maximale du capital (%)", 10, 100, 40)
    risk_pct = st.slider("Risque théorique par trade (%)", 0.1, 2.0, 0.5, 0.1)
    stop_pct = st.slider("Stop-loss (%)", 0.3, 3.0, 1.0, 0.1)
    target_pct = st.slider("Objectif de gain (%)", 0.5, 5.0, 1.5, 0.1)
    daily_loss_pct = st.slider("Arrêt journalier si perte (%)", 0.5, 5.0, 1.5, 0.1)

@st.cache_data(ttl=300)
def get_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in needed if c in df.columns]].dropna()
    return df

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def simulate(data, initial_cash, fee, spread, slip, exposure_pct,
             risk_pct, stop_pct, target_pct, daily_loss_pct):
    d = data.copy()
    d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    d["RSI"] = rsi(d["Close"])
    d["VolMA20"] = d["Volume"].rolling(20).mean()
    d = d.dropna().copy()

    cash = float(initial_cash)
    qty = 0.0
    entry = 0.0
    entry_time = None
    entry_cost = 0.0
    stop = target = 0.0
    trades = []
    equity_curve = []
    day_start_equity = initial_cash
    current_day = None
    trading_paused = False
    round_trip_cost_pct = (spread + slip) / 100

    for ts, row in d.iterrows():
        price = float(row["Close"])
        day = ts.date()
        if current_day != day:
            current_day = day
            day_start_equity = cash + qty * price
            trading_paused = False

        # Mark-to-market equity
        equity = cash + qty * price
        equity_curve.append((ts, equity))

        # Daily loss circuit breaker
        if equity <= day_start_equity * (1 - daily_loss_pct / 100):
            trading_paused = True

        if qty > 0:
            # Conservative intrabar handling: if both levels touched, assume stop first.
            low = float(row["Low"])
            high = float(row["High"])
            exit_reason = None
            raw_exit = None
            if low <= stop:
                exit_reason, raw_exit = "Stop-loss", stop
            elif high >= target:
                exit_reason, raw_exit = "Objectif", target
            elif entry_time is not None and (ts - entry_time).total_seconds() >= 6 * 3600:
                exit_reason, raw_exit = "Fin de séance / durée max", price

            if exit_reason:
                effective_exit = raw_exit * (1 - round_trip_cost_pct)
                gross_proceeds = qty * effective_exit
                exit_fee = fee
                cash += gross_proceeds - exit_fee
                net_pnl = cash - (cash - (gross_proceeds - exit_fee))  # proceeds net of exit costs
                # Reconstruct trade P&L from original outlay and net sale proceeds.
                total_outlay = entry * qty * (1 + round_trip_cost_pct) + fee
                net_sale = gross_proceeds - exit_fee
                pnl = net_sale - total_outlay
                trades.append({
                    "Entrée": entry_time, "Sortie": ts, "Prix achat": entry,
                    "Prix vente": raw_exit, "Quantité": qty, "Motif": exit_reason,
                    "Résultat net (€)": pnl, "Rendement net (%)": 100*pnl/total_outlay
                })
                qty = 0.0
                entry = 0.0
                entry_time = None
                continue

        if qty == 0 and not trading_paused:
            # Buy a pullback/rebound only in an established short-term uptrend.
            prev = d.shift(1).loc[ts]
            signal = (
                row["Close"] > row["EMA21"] and
                row["EMA9"] > row["EMA21"] and
                prev["Close"] <= prev["EMA9"] and
                row["Close"] > row["EMA9"] and
                42 <= row["RSI"] <= 68 and
                row["Volume"] >= 0.8 * row["VolMA20"]
            )
            if signal:
                # Position size is limited by both risk budget and max exposure.
                risk_budget = equity * risk_pct / 100
                risk_based_value = risk_budget / (stop_pct / 100)
                position_value = min(equity * exposure_pct / 100, risk_based_value, cash * 0.98)
                effective_entry = price * (1 + round_trip_cost_pct)
                units = (position_value - fee) / effective_entry if position_value > fee else 0
                if units > 0:
                    required = units * effective_entry + fee
                    if required <= cash:
                        cash -= required
                        qty = units
                        entry = price
                        entry_time = ts
                        entry_cost = required
                        stop = entry * (1 - stop_pct / 100)
                        target = entry * (1 + target_pct / 100)

    # Close any open position at last close for a fair end-of-test valuation.
    if qty > 0:
        ts = d.index[-1]
        raw_exit = float(d.iloc[-1]["Close"])
        effective_exit = raw_exit * (1 - round_trip_cost_pct)
        net_sale = qty * effective_exit - fee
        total_outlay = entry * qty * (1 + round_trip_cost_pct) + fee
        pnl = net_sale - total_outlay
        cash += net_sale
        trades.append({
            "Entrée": entry_time, "Sortie": ts, "Prix achat": entry,
            "Prix vente": raw_exit, "Quantité": qty, "Motif": "Fin historique",
            "Résultat net (€)": pnl, "Rendement net (%)": 100*pnl/total_outlay
        })
        qty = 0

    trades_df = pd.DataFrame(trades)
    curve = pd.DataFrame(equity_curve, columns=["Date", "Capital"]).set_index("Date")
    final_value = cash
    return d, trades_df, curve, final_value

if st.button("Lancer le backtest", type="primary"):
    try:
        data = get_data(ticker.strip(), period, interval)
        if len(data) < 80:
            st.error("Historique insuffisant. Essayez une autre période ou un autre intervalle.")
            st.stop()
        tested, trades, curve, final_value = simulate(
            data, capital, fee, spread_pct, slippage_pct, max_exposure,
            risk_pct, stop_pct, target_pct, daily_loss_pct
        )
        pnl = final_value - capital
        ret = 100 * pnl / capital
        wins = int((trades["Résultat net (€)"] > 0).sum()) if not trades.empty else 0
        losses = int((trades["Résultat net (€)"] <= 0).sum()) if not trades.empty else 0
        win_rate = 100*wins/len(trades) if len(trades) else 0
        st.subheader(f"Résultats — {ticker.upper()}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital final", f"{final_value:,.2f} €".replace(",", " "))
        c2.metric("Résultat net", f"{pnl:+,.2f} €".replace(",", " "), f"{ret:+.2f} %")
        c3.metric("Transactions clôturées", len(trades))
        c4.metric("Trades gagnants", f"{win_rate:.1f} %", f"{wins} gagnants / {losses} perdants")
        st.line_chart(curve["Capital"])
        if not trades.empty:
            st.subheader("Historique des opérations")
            st.dataframe(trades, use_container_width=True)
            st.download_button("Télécharger les trades (CSV)", trades.to_csv(index=False).encode("utf-8"), "trades_backtest.csv", "text/csv")
        else:
            st.info("Aucun signal d'achat sur la période : c'est possible et préférable à des transactions forcées.")
        st.caption("Les performances passées simulées ne préjugent pas des performances futures. Le modèle de spread/glissement est une approximation. Fiscalité non incluse.")
    except Exception as e:
        st.error(f"Erreur de récupération ou de simulation : {e}")

st.divider()
st.subheader("À savoir avant toute utilisation réelle")
st.markdown("""
- Trade Republic facture actuellement **1 € par ordre ponctuel exécuté**, donc environ **2 € par aller-retour**, auxquels s'ajoutent spread et éventuels coûts tiers.
- Trade Republic ne publie pas d'API officielle de trading pour les particuliers. Ce prototype n'envoie donc aucun ordre réel.
- Les données Yahoo Finance servent à tester une stratégie, pas à garantir une exécution en temps réel au prix affiché.
- Le backtest est volontairement long-only, sans levier, avec une seule position à la fois et un coupe-circuit journalier.
""")
