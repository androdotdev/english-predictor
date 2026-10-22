import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(mo):
    mo.md("""
    ## Data Cleaning And Pre-processing
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("./data/cefr_leveled_texts.csv")
    df = df.drop_duplicates(subset="text")
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df["label"].value_counts()
    return


@app.cell
def _(df):
    df["text"].duplicated().sum()
    return


@app.cell
def _(df):
    df["text"].str.len().describe()
    return


@app.cell
def _(df):
    df.groupby("label")["text"].apply(lambda x: x.str.len().mean())
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md("""
    ## DATA SPLITTING
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split

    return (train_test_split,)


@app.cell
def _(df, train_test_split):
    x = df["text"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=.2,
        random_state=42,
        stratify=df["label"]
    )
    return x_test, x_train, y_test, y_train


@app.cell
def _(mo):
    mo.md("""
    ## Vectorizing Text
    """)
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return (TfidfVectorizer,)


@app.cell
def _(TfidfVectorizer, x_test, x_train):
    vectorizer = TfidfVectorizer()

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    return x_test_vectorized, x_train_vectorized


@app.cell
def _(mo):
    mo.md("""
    ## Decision Tree Classifier
    """)
    return


@app.cell
def _():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    return DecisionTreeClassifier, accuracy_score


@app.cell
def _(DecisionTreeClassifier, x_test_vectorized, x_train_vectorized, y_train):
    decision_model = DecisionTreeClassifier(random_state=42)
    decision_model.fit(x_train_vectorized, y_train)
    decision_y_pred = decision_model.predict(x_test_vectorized)
    return (decision_y_pred,)


@app.cell
def _(mo):
    mo.md("""
    ### Plotting Graph
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    return confusion_matrix, plt, sns


@app.cell
def _(confusion_matrix, decision_y_pred, plt, sns, y_test):
    cm_dt = confusion_matrix(y_test, decision_y_pred, labels=["A1", "A2", "B1", "B2", "C1", "C2"])

    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        xticklabels=["A1", "A2", "B1", "B2", "C1", "C2"],
        yticklabels=["A1", "A2", "B1", "B2", "C1", "C2"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Decision Tree Confusion Matrix")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ### Metrics
    """)
    return


@app.cell
def _(accuracy_score, decision_y_pred, y_test):
    accuracy_dt = accuracy_score(y_test, decision_y_pred)
    accuracy_dt
    return


@app.cell
def _(decision_y_pred, y_test):
    from sklearn.metrics import classification_report
    print(classification_report(y_test, decision_y_pred))
    return (classification_report,)


@app.cell
def _(mo):
    mo.md("""
    ## KKN Classifier
    """)
    return


@app.cell
def _():
    from sklearn.neighbors import KNeighborsClassifier

    return (KNeighborsClassifier,)


@app.cell
def _(KNeighborsClassifier, x_test_vectorized, x_train_vectorized, y_train):
    kkn = KNeighborsClassifier(n_neighbors=5)
    kkn_model = kkn.fit(x_train_vectorized, y_train)
    kkn_y_predict = kkn_model.predict(x_test_vectorized)
    return (kkn_y_predict,)


@app.cell
def _(mo):
    mo.md("""
    ### Plotting Graph
    """)
    return


@app.cell
def _(confusion_matrix, kkn_y_predict, plt, sns, y_test):
    cm_kkn = confusion_matrix(y_test, kkn_y_predict, labels=["A1", "A2", "B1", "B2", "C1", "C2"])

    sns.heatmap(
        cm_kkn,
        annot=True,
        fmt="d",
        xticklabels=["A1", "A2", "B1", "B2", "C1", "C2"],
        yticklabels=["A1", "A2", "B1", "B2", "C1", "C2"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("KKN Confusion Matrix")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ### Metrics
    """)
    return


@app.cell
def _(accuracy_score, decision_y_pred, y_test):
    accuracy_kkn = accuracy_score(y_test, decision_y_pred)
    accuracy_kkn
    return


@app.cell
def _(classification_report, kkn_y_predict, y_test):
    print(classification_report(y_test, kkn_y_predict))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Random Forest
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## LinearSVC
    """)
    return


@app.cell
def _():
    from sklearn.svm import LinearSVC

    return (LinearSVC,)


@app.cell
def _(LinearSVC, x_test_vectorized, x_train_vectorized, y_train):
    linear_model = LinearSVC(class_weight='balanced', C=1.0, random_state=42)
    linear_model.fit(x_train_vectorized, y_train)
    linear_y_pred = linear_model.predict(x_test_vectorized)
    return (linear_y_pred,)


@app.cell
def _(classification_report, confusion_matrix, linear_y_pred, y_test):
    print("--- New Confusion Matrix ---")
    print(confusion_matrix(y_test, linear_y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, linear_y_pred))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
