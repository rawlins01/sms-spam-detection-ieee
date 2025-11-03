# pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

SEED = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv("data/sms_spam_collection.csv", sep="\t", header=None, names=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].str.lower()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
)

# TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, norm='l2')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
models = {}

# Logistic Regression
lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=SEED)
lr.fit(X_train_tfidf, y_train)
models["Logistic Regression"] = lr

# Calibrated SVM
svm = LinearSVC(C=1.0, class_weight='balanced', random_state=SEED)
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
calibrated_svm.fit(X_train_tfidf, y_train)
models["Linear SVM (calibrated)"] = calibrated_svm

# Naive Bayes
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_tfidf, y_train)
models["Multinomial Naive Bayes"] = nb

# Evaluate
results = []
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    else:
        y_prob = model.decision_function(X_test_tfidf)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # normalize
    y_pred = (y_prob >= 0.5).astype(int)
    pr_auc = auc(*precision_recall_curve(y_test, y_prob)[1::-1])
    f1 = f1_score(y_test, y_pred)
    precision = (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
    recall = (y_pred & y_test).sum() / y_test.sum()
    results.append({"Model": name, "PR-AUC": round(pr_auc, 4), "F1": round(f1, 4),
                    "Precision": round(precision, 4), "Recall": round(recall, 4)})

# Print results
print("\nResults:")
print(pd.DataFrame(results))

# Save PR curve
y_prob_svm = models["Linear SVM (calibrated)"].predict_proba(X_test_tfidf)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob_svm)
plt.figure(); plt.plot(recall, precision); plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('PR Curve (SVM)'); plt.savefig(f"{OUTPUT_DIR}/pr_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# Save confusion matrix
cm = confusion_matrix(y_test, (y_prob_svm >= 0.5).astype(int))
plt.figure(); plt.imshow(cm, cmap='Purples'); plt.colorbar()
for i in range(2): 
    for j in range(2): 
        plt.text(j, i, cm[i,j], ha='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.xticks([0,1], ['Ham', 'Spam']); plt.yticks([0,1], ['Ham', 'Spam'])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# Save model
joblib.dump(models["Linear SVM (calibrated)"], "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"\nOutputs saved to {OUTPUT_DIR}/")
