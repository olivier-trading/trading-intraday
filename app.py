
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# ============================================================
# SIMULATEUR BOURSIER V2 - ARGENT STRICTEMENT FICTIF
# AUCUNE CONNEXION A UN COURTIER - AUCUN ORDRE REEL
# ============================================================

st.set_page_config(
    page_title="Investisseur intelligent V2",
    page_icon="📈",
    layout="wide"
)

INITIAL_CAPITAL = 500.0
MAX_POSITIONS = 10
MIN_ORDER = 30.0
REFRESH_MS = 4 * 60 * 60 * 1000 + 48 * 60 * 1000

# Entreprises sélectionnées : grandes sociétés établies.
# Les cours sont téléchargés depuis Yahoo Finance.
STOCKS = {
    "AI.PA": "Air Liquide",
    "SAF.PA": "Safran",
    "SU.PA": "Schneider Electric",
    "AIR.PA": "Airbus",
    "OR.PA": "L'Oréal",
    "MC.PA": "LVMH",
    "SAN.PA": "Sanofi",
    "BNP.PA": "BNP Paribas",
    "TTE.PA": "TotalEnergies",
    "DG.PA": "Vinci",
    "CAP.PA": "Capgemini",
    "ACA.PA": "Crédit Agricole",
    "ENGI.PA": "Engie",
    "KER.PA": "Kering",
    "RMS.PA": "Hermès",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Alphabet",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "META": "Meta",
    "AVGO": "Broadcom",
    "JPM": "JPMorgan",
    "LLY": "Eli Lilly",
    "COST": "Costco",
    "ORCL": "Oracle",
    "CAT": "Caterpillar",
    "AMD": "AMD",
    "SAP.DE": "SAP",
    "SIE.DE": "Siemens"
}

# ============================================================
# INITIALISATION DU PORTEFEUILLE FICTIF
# ============================================================

if "cash" not in st.session_state:
    st.session_state.cash = INITIAL_CAPITAL
    st.session_state.positions = {}
    st.session_state.history = []
    st.session_state.initialized = True
    st.session_state.last_bar = None
    st.session_state.total_fees = 0.0

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def analyze_stock(df):
    if df is None or len(df) < 110:
        return None

    close = df["Close"].dropna()

    if len(close) < 110:
        return None

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    rsi = calculate_rsi(close)

    price = float(close.iloc[-1])
    previous_price = float(close.iloc[-2])

    high100 = float(close.tail(100).max())
    correction = (price / high100 - 1) * 100

    # Qualité de tendance
    trend_score = 0

    if price > sma50.iloc[-1]:
        trend_score += 10

    if sma50.iloc[-1] > sma100.iloc[-1]:
        trend_score += 15

    if sma20.iloc[-1] > sma50.iloc[-1]:
        trend_score += 10

    # Recherche d'une correction raisonnable
    correction_score = 0

    if -12 <= correction <= -3:
        correction_score = 25
    elif -15 <= correction < -12:
        correction_score = 10

    # Recherche d'un début de reprise
    momentum_score = 0

    if price > previous_price:
        momentum_score += 10

    if close.iloc[-1] > sma20.iloc[-1]:
        momentum_score += 10

    current_rsi = float(rsi.iloc[-1])

    if 40 <= current_rsi <= 65:
        momentum_score += 10

    # Liquidité et activité
    volume_score = 0

    if "Volume" in df.columns:
        volumes = df["Volume"].dropna()

        if len(volumes) >= 21:
            average_volume = volumes.tail(21).mean()

            if average_volume > 0:
                if volumes.iloc[-1] >= average_volume * 1.1:
                    volume_score = 10
                else:
                    volume_score = 5

    score = (
        trend_score
        + correction_score
        + momentum_score
        + volume_score
    )

    return {
        "price": price,
        "score": min(score, 100),
        "correction": correction,
        "rsi": current_rsi,
        "sma20": float(sma20.iloc[-1]),
        "sma50": float(sma50.iloc[-1]),
        "sma100": float(sma100.iloc[-1]),
        "timestamp": close.index[-1]
    }


# ============================================================
# TELECHARGEMENT DES COURS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def download_market_data():
    tickers = list(STOCKS.keys()) + ["EURUSD=X"]

    data = yf.download(
        tickers,
        period="3mo",
        interval="1h",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True
    )

    results = {}

    for ticker in STOCKS:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[ticker].copy()
            else:
                continue

            df = df.dropna(subset=["Close"])

            analysis = analyze_stock(df)

            if analysis:
                results[ticker] = analysis

        except Exception:
            continue

    # Conversion USD vers EUR
    try:
        fx = data["EURUSD=X"]["Close"].dropna()
        eurusd = float(fx.iloc[-1])
    except Exception:
        eurusd = None

    if eurusd and eurusd > 0:
        for ticker in results:
            if ticker in [
                "MSFT", "NVDA", "GOOGL", "AAPL", "AMZN",
                "META", "AVGO", "JPM", "LLY", "COST",
                "ORCL", "CAT", "AMD"
            ]:
                results[ticker]["price"] /= eurusd

    return results


# ============================================================
# CALCUL DU PORTEFEUILLE
# ============================================================

def portfolio_value(market):
    value = st.session_state.cash

    for ticker, pos in st.session_state.positions.items():
        if ticker in market:
            value += (
                pos["quantity"] * market[ticker]["price"]
            )
        else:
            value += pos["quantity"] * pos["last_price"]

    return value


def execute_buy(ticker, analysis, budget, fee_rate):
    price = analysis["price"]

    fee = budget * fee_rate
    invested = budget - fee

    quantity = invested / price

    st.session_state.cash -= budget
    st.session_state.total_fees += fee

    st.session_state.positions[ticker] = {
        "name": STOCKS[ticker],
        "quantity": quantity,
        "buy_price": price,
        "last_price": price,
        "peak_price": price,
        "buy_date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "score": analysis["score"],
        "invested": budget
    }

    st.session_state.history.append({
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Entreprise": STOCKS[ticker],
        "Opération": "ACHAT",
        "Prix EUR": round(price, 2),
        "Montant EUR": round(budget, 2),
        "Résultat EUR": 0.0,
        "Frais EUR": round(fee, 2)
    })


def execute_sell(ticker, price, reason, fee_rate):
    pos = st.session_state.positions[ticker]

    gross = pos["quantity"] * price
    fee = gross * fee_rate
    net = gross - fee

    result = net - pos["invested"]

    st.session_state.cash += net
    st.session_state.total_fees += fee

    st.session_state.history.append({
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Entreprise": pos["name"],
        "Opération": "VENTE - " + reason,
        "Prix EUR": round(price, 2),
        "Montant EUR": round(net, 2),
        "Résultat EUR": round(result, 2),
        "Frais EUR": round(fee, 2)
    })

    del st.session_state.positions[ticker]


# ============================================================
# INTERFACE
# ============================================================

st.title("📈 Investisseur intelligent – Version 2")

st.caption(
    "SIMULATION UNIQUEMENT : argent fictif, aucun ordre réel."
)

st_autorefresh(
    interval=REFRESH_MS,
    key="automatic_refresh"
)

with st.sidebar:
    st.header("Paramètres")

    fee_rate = st.number_input(
        "Frais par transaction (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.10,
        step=0.05
    ) / 100

    stop_loss = st.slider(
        "Stop-loss (%)",
        2.0, 12.0, 5.0, 0.5
    ) / 100

    trailing_stop = st.slider(
        "Stop suiveur (%)",
        2.0, 12.0, 6.0, 0.5
    ) / 100

    take_profit = st.slider(
        "Objectif de bénéfice (%)",
        3.0, 20.0, 8.0, 0.5
    ) / 100

    min_score = st.slider(
        "Score minimum pour acheter",
        50, 95, 75
    )

    st.divider()

    st.write("Capital initial : 500 €")
    st.write("Positions maximales : 10")
    st.write("Entreprises surveillées :", len(STOCKS))

    automatic = st.toggle(
        "Décisions fictives automatiques",
        value=True
    )

    if st.button("Réinitialiser le portefeuille"):
        st.session_state.cash = INITIAL_CAPITAL
        st.session_state.positions = {}
        st.session_state.history = []
        st.session_state.last_bar = None
        st.session_state.total_fees = 0.0
        st.rerun()

# ============================================================
# RECUPERATION DES DONNEES
# ============================================================

if st.button("🔄 Actualiser les cours maintenant"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Analyse des 30 entreprises en cours..."):
    market = download_market_data()

if not market:
    st.error(
        "Impossible de récupérer les cours. "
        "Vérifie ta connexion ou réessaie plus tard."
    )
    st.stop()

# ============================================================
# DECISIONS AUTOMATIQUES - SIMULATION UNIQUEMENT
# ============================================================

new_timestamps = [
    str(x["timestamp"]) for x in market.values()
]

current_bar = max(new_timestamps) if new_timestamps else None

new_market_bar = (
    current_bar is not None
    and current_bar != st.session_state.last_bar
)

if automatic and new_market_bar:

    # 1. Gestion des positions existantes
    for ticker in list(st.session_state.positions.keys()):

        if ticker not in market:
            continue

        pos = st.session_state.positions[ticker]
        price = market[ticker]["price"]

        pos["last_price"] = price
        pos["peak_price"] = max(pos["peak_price"], price)

        variation = price / pos["buy_price"] - 1
        drawdown = price / pos["peak_price"] - 1

        reason = None

        if variation <= -stop_loss:
            reason = "STOP LOSS"

        elif drawdown <= -trailing_stop and variation > 0:
            reason = "STOP SUIVEUR"

        elif variation >= take_profit:
            reason = "OBJECTIF ATTEINT"

        elif market[ticker]["score"] < 35:
            reason = "DÉGRADATION DU SIGNAL"

        if reason:
            execute_sell(
                ticker, price, reason, fee_rate
            )

    # 2. Recherche de nouvelles opportunités
    candidates = []

    for ticker, analysis in market.items():

        if ticker in st.session_state.positions:
            continue

        if analysis["score"] >= min_score:
            candidates.append((ticker, analysis))

    candidates.sort(
        key=lambda x: x[1]["score"],
        reverse=True
    )

    # 3. Achat des meilleures opportunités
    for ticker, analysis in candidates:

        if len(st.session_state.positions) >= MAX_POSITIONS:
            break

        portfolio = portfolio_value(market)

        remaining_slots = (
            MAX_POSITIONS - len(st.session_state.positions)
        )

        budget = min(
            st.session_state.cash,
            portfolio / MAX_POSITIONS
        )

        if budget < MIN_ORDER:
            break

        if st.session_state.cash < MIN_ORDER:
            break

        execute_buy(
            ticker,
            analysis,
            budget,
            fee_rate
        )

    st.session_state.last_bar = current_bar

# ============================================================
# MISE A JOUR DES PRIX DES POSITIONS
# ============================================================

for ticker, pos in st.session_state.positions.items():
    if ticker in market:
        pos["last_price"] = market[ticker]["price"]

# ============================================================
# TABLEAU DE BORD
# ============================================================

total = portfolio_value(market)

profit = total - INITIAL_CAPITAL

invested = sum(
    p["invested"]
    for p in st.session_state.positions.values()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Capital total",
    f"{total:.2f} €",
    f"{profit:+.2f} €"
)

c2.metric(
    "Liquidités disponibles",
    f"{st.session_state.cash:.2f} €"
)

c3.metric(
    "Positions ouvertes",
    len(st.session_state.positions)
)

c4.metric(
    "Frais cumulés",
    f"{st.session_state.total_fees:.2f} €"
)

st.divider()

# ============================================================
# ANALYSE DES ENTREPRISES
# ============================================================

st.subheader("1. Classement des entreprises")

rows = []

for ticker, a in market.items():

    if a["score"] >= min_score:
        status = "ACHAT POTENTIEL"
    elif a["score"] >= 60:
        status = "SURVEILLER"
    else:
        status = "ÉCARTER"

    rows.append({
        "Entreprise": STOCKS[ticker],
        "Ticker": ticker,
        "Cours (€)": round(a["price"], 2),
        "Score": a["score"],
        "Correction (%)": round(a["correction"], 2),
        "RSI": round(a["rsi"], 1),
        "Décision": status
    })

ranking = pd.DataFrame(rows).sort_values(
    "Score", ascending=False
)

st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# POSITIONS OUVERTES
# ============================================================

st.subheader("2. Positions ouvertes")

if st.session_state.positions:

    open_rows = []

    for ticker, pos in st.session_state.positions.items():

        price = pos["last_price"]

        value = pos["quantity"] * price
        result = value - pos["invested"]

        open_rows.append({
            "Entreprise": pos["name"],
            "Prix achat (€)": round(pos["buy_price"], 2),
            "Cours actuel (€)": round(price, 2),
            "Investissement (€)": round(pos["invested"], 2),
            "Valeur actuelle (€)": round(value, 2),
            "Résultat latent (€)": round(result, 2),
            "Variation (%)": round(
                (price / pos["buy_price"] - 1) * 100, 2
            )
        })

    st.dataframe(
        pd.DataFrame(open_rows),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("Aucune position ouverte.")

# ============================================================
# HISTORIQUE DES OPERATIONS
# ============================================================

st.subheader("3. Historique des transactions")

history = pd.DataFrame(st.session_state.history)

if not history.empty:

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    realized = history.loc[
        history["Opération"].str.startswith("VENTE"),
        "Résultat EUR"
    ].sum()

    st.metric(
        "Résultat cumulé des ventes réalisées",
        f"{realized:+.2f} €"
    )

    csv = history.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger l'historique CSV",
        csv,
        file_name="historique_simulation.csv",
        mime="text/csv"
    )

else:
    st.info("Aucune transaction réalisée.")

# ============================================================
# RESULTAT GLOBAL
# ============================================================

st.divider()
st.subheader("4. Bilan de la stratégie")

history = pd.DataFrame(st.session_state.history)

if not history.empty:
    sales = history[
        history["Opération"].str.startswith("VENTE")
    ]

    winners = (
        sales["Résultat EUR"] > 0
    ).sum()

    losers = (
        sales["Résultat EUR"] < 0
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Ventes gagnantes", int(winners))
    col2.metric("Ventes perdantes", int(losers))

    win_rate = (
        winners / len(sales) * 100
        if len(sales) else 0
    )

    col3.metric("Taux de réussite", f"{win_rate:.1f} %")

st.write(
    "Les performances affichées sont fictives et ne constituent "
    "pas une prévision de rendement réel."
)

st.caption(
    "Données de marché fournies par Yahoo Finance. "
    "Les cours peuvent être retardés ou indisponibles. "
    "Les décisions sont simulées et les données du portefeuille "
    "sont conservées dans la session de l'application."
)
