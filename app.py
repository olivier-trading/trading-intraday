
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
        try:
            if isinstance(change.columns, pd.MultiIndex):
                close = None

                for niveau in range(change.columns.nlevels):
                    if "Close" in change.columns.get_level_values(niveau):
                        close = change.xs(
                            "Close",
                            axis=1,
                            level=niveau
                        )
                        break
            else:
                close = change["Close"]

            if close is not None:
                if isinstance(close, pd.DataFrame):
                    valeurs = close.stack().dropna()
                else:
                    valeurs = close.dropna()

                if not valeurs.empty:
                    taux = float(valeurs.iloc[-1])

        except Exception:
            taux = None

    return quotidien, horaire, taux
