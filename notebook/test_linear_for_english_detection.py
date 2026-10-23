import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import re
    import numpy as np
    import pandas as pd
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return (
        LinearSVC,
        StandardScaler,
        TfidfVectorizer,
        hstack,
        np,
        pd,
        re,
        train_test_split,
    )


@app.cell
def _(np, re):
    # 1. Custom Grammar & Structure Extractor
    def extract_linguistic_features(text_series):
        features = []
        for text in text_series:
            text = str(text)
            words = text.split()
            # Split sentences by terminal punctuation
            sentences = [s for s in re.split(r"[.!?]", text) if len(s.strip()) > 0]

            word_count = len(words) if len(words) > 0 else 1
            sent_count = len(sentences) if len(sentences) > 0 else 1

            # 1. Basic Stats
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            avg_sent_len = word_count / sent_count

            # 2. Sentence Length Variance (Advanced text mixes short and very long sentences)
            sent_lengths = [len(s.split()) for s in sentences]
            sent_len_var = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

            # 3. Structural Complexity Markers
            long_words = sum(1 for w in words if len(w) > 6) / word_count

            # Punctuation density (Commas/semicolons indicate dependent clauses)
            punctuation_density = (
                len(re.findall(r"[,;:\-–—]", text)) / sent_count
            )

            # 4. B1/B2/C1 Functional Transition Words
            # B1 relies on simple connectors (but, so, because)
            # B2/C1 relies on complex connectors (however, furthermore, despite, although)
            complex_connectors = len(
                re.findall(
                    r"\b(however|furthermore|moreover|nevertheless|although|despite|whereas|consequently|therefore)\b",
                    text,
                    flags=re.IGNORECASE,
                    )
            ) / sent_count

            features.append(
                [
                    avg_word_len,
                    avg_sent_len,
                    sent_len_var,
                    long_words,
                    punctuation_density,
                    complex_connectors,
                ]
            )

        return np.array(features)



    return (extract_linguistic_features,)


@app.cell
def _(TfidfVectorizer, pd, train_test_split):

    # 2. Load Dataset
    df = pd.read_csv("data/cefr_leveled_texts.csv")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # 3. Double Vectorizer Setup (Words + Word Parts)
    # Word vectorizer captures vocabulary
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
    # Char vectorizer captures advanced suffixes/prefixes
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=4000
    )
    return (
        X_test_text,
        X_train_text,
        char_vectorizer,
        word_vectorizer,
        y_test,
        y_train,
    )


@app.cell
def _(
    StandardScaler,
    X_test_text,
    X_train_text,
    char_vectorizer,
    extract_linguistic_features,
    hstack,
    word_vectorizer,
):

    # Fit and Transform text
    X_train_word = word_vectorizer.fit_transform(X_train_text)
    X_test_word = word_vectorizer.transform(X_test_text)

    X_train_char = char_vectorizer.fit_transform(X_train_text)
    X_test_char = char_vectorizer.transform(X_test_text)

    # 4. Extract and Scale Linguistic Features
    X_train_ling = extract_linguistic_features(X_train_text)
    X_test_ling = extract_linguistic_features(X_test_text)

    # SVM requires scaled numerical features
    scaler = StandardScaler()
    X_train_ling_scaled = scaler.fit_transform(X_train_ling)
    X_test_ling_scaled = scaler.transform(X_test_ling)

    # 5. Stack Everything Together
    X_train_final = hstack([X_train_word, X_train_char, X_train_ling_scaled])
    X_test_final = hstack([X_test_word, X_test_char, X_test_ling_scaled])
    return X_test_final, X_train_final


@app.cell
def _(LinearSVC, X_test_final, X_train_final, y_test, y_train):

    # 6. Train LinearSVC with balanced class weights
    # class_weight='balanced' forces the model to pay extra attention to harder, rarer levels
    model = LinearSVC(C=0.2, class_weight="balanced", random_state=42)
    model.fit(X_train_final, y_train)

    # 7. Evaluate
    print(f"New LinearSVC Accuracy: {model.score(X_test_final, y_test):.2f}")
    return (model,)


@app.cell
def _(X_test_final, model):
    y_pred = model.predict(X_test_final)
    return (y_pred,)


@app.cell
def _(y_pred, y_test):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    print("---  Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _(char_vectorizer, model, word_vectorizer):
    import joblib

    joblib.dump(word_vectorizer, "model/word_vectorizer.joblib")
    joblib.dump(char_vectorizer, "model/char_vectorizer.joblib")
    joblib.dump(model, "model/cefr_model.joblib")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
