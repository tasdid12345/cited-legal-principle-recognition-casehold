# Recognizing cited legal principles: TF-IDF + overlap features vs Naive Bayes

## Motivation
The paper "Recognizing cited facts and principles in legal judgements" (2017) classifies statements in judgements as cited facts or principles using n-gram based classifiers. This experiment asks whether a stronger feature set and imbalance handling improve on such an n-gram Naive Bayes baseline.

## Question
Can a tuned TF-IDF (word plus character n-gram) model with context-candidate overlap features and class-imbalance handling beat a Naive Bayes n-gram baseline at recognizing which statements are genuine cited legal principles (holdings)?

## Dataset
CaseHOLD from LexGLUE (Hugging Face, parquet over HTTP). Each citing context has one true holding and four distractors; each candidate is a binary instance. A subsample of the train split is used for training and a subsample of the official test split as the held-out set. This is **not** the paper's own corpus (annotated UK/BAILII judgements, not freely available); it is a public proxy.

## Models
- **Baseline:** multinomial Naive Bayes on word uni/bigram counts of the candidate.
- **Improved:** logistic regression on word and character TF-IDF of the candidate plus context-candidate overlap features, with class weighting and a small context-grouped cross-validated search (train only).

Main metric: F1 of the true-principle class. Secondary: top-1 accuracy per context, ROC-AUC. Decision thresholds are chosen from out-of-fold train predictions; the test set is used once.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

## Results (latest run)

On a CaseHOLD proxy test set (US case law, not BAILII), my re-implementation of your approach reached 37% F1 on the true-principle class with a Naive Bayes baseline, while my TF-IDF plus overlap-feature logistic regression reached 59% F1 on the same held-out test set.

| Metric | Value |
|---|---|
| baseline_score | 0.366 |
| improved_score | 0.5891 |
| improvement | 0.2231 |
| baseline_f1 | 0.366 |
| improved_f1 | 0.5891 |
| baseline_top1_acc | 0.284 |
| improved_top1_acc | 0.5973 |
| baseline_roc_auc | 0.6181 |
| improved_roc_auc | 0.855 |
| random_top1_acc | 0.2 |

Run on 2026-10-10. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Proxy dataset (US case law, not UK BAILII) so results are not comparable to the paper. Both models use an F1-optimal threshold picked from out-of-fold train scores. TF-IDF vocabularies are fit unsupervised on train candidates before CV. Best params: baseline {'multinomialnb__alpha': 0.5}, improved {'C': 0.3, 'class_weight': None}. Single run, single seed, no confidence intervals; the test set is subsampled.
