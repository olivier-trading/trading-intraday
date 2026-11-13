
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
        progress=False,
        threads=True
    )

    taux = None

    try:
        if change is not None and not change.empty:

            close = None

            if isinstance(change.columns, pd.MultiIndex):

                for niveau in range(change.columns.nlevels):
                    valeurs_niveau = change.columns.get_level_values(niveau)

                    if "Close" in valeurs_niveau:
                        close = change.xs(
                            "Close",
                            axis=1,
                            level=niveau
                        )
                        break

            else:
                if "Close" in change.columns:
                    close = change["Close"]

            if close is not None:

                if isinstance(close, pd.DataFrame):
                    valeurs = close.iloc[:, 0].dropna()
                elif isinstance(close, pd.Series):
                    valeurs = close.dropna()
                else:
                    valeurs = pd.Series(dtype=float)

                if not valeurs.empty:
                    taux = float(valeurs.iloc[-1])

    except Exception as e:
        print(f"Erreur récupération taux EUR/USD : {e}")
        taux = None

    return quotidien, horaire, taux
