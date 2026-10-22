import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("./data/creditcard.csv")
    df
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
