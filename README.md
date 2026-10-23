# 📚 English CEFR Predictor

A machine learning project that predicts the **CEFR English writing level** of a text sample.

The project compares multiple classification algorithms using TF-IDF text features and evaluates their performance across six CEFR levels:

**A1, A2, B1, B2, C1, C2**

## 🎯 Objective

The goal is to investigate how well traditional machine learning algorithms can classify English writing according to CEFR level based only on the written text.

The project was developed as part of a classification algorithm evaluation exercise.

## 📊 Dataset

The dataset contains English writing samples labelled with CEFR levels:

- A1
- A2
- B1
- B2
- C1
- C2

The main input feature is the **text itself**, while the target variable is the corresponding CEFR level.

Because the model primarily relies on textual information, the project also investigates some of the limitations of using raw text as the only feature.

## 🔧 Preprocessing

The text was converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The vectorizer was fitted only on the training data:

```python
vectorizer = TfidfVectorizer()

x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)
```

This prevents information from the test set from being used during training.

## 🤖 Models

Three classification algorithms were evaluated:

1. **Linear Support Vector Classifier (LinearSVC)**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**

All three models were evaluated using the same TF-IDF representation to make the comparison fair.

## 📈 Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **Linear SVC** | **0.59** | **0.57** | **0.58** |
| KNN | 0.48 | 0.46 | 0.48 |
| Decision Tree | 0.41 | 0.40 | 0.41 |

### Linear SVC — Per-Class Performance

| CEFR Level | Precision | Recall | F1 |
|---|---:|---:|---:|
| A1 | 0.76 | 0.88 | 0.82 |
| A2 | 0.69 | 0.65 | 0.67 |
| B1
