"""CaseHOLD (LexGLUE) proxy: NB n-gram baseline vs tuned TF-IDF + overlap-feature logistic regression."""
import io, json, os, re, sys, time
import numpy as np, pandas as pd, requests, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, N_TRAIN_CTX, N_TEST_CTX, K = 42, 6000, 1500, 5
np.random.seed(SEED)
API = "https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/case_hold/%s"
RAW = "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/refs%2Fconvert%2Fparquet/case_hold/{s}/0000.parquet"

def get(url):  # GET with timeout and backoff on 429/5xx
    for i in range(5):
        r = requests.get(url, timeout=60)
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status(); return r
        time.sleep(2 ** i * 2)
    r.raise_for_status()

def load_split(split, n_ctx):
    try: urls = get(API % split).json()
    except Exception: urls = [RAW.format(s=split)]
    parts = []
    for u in urls:  # read shards until enough contexts
        parts.append(pd.read_parquet(io.BytesIO(get(u).content)))
        if sum(map(len, parts)) >= n_ctx: break
    d = pd.concat(parts, ignore_index=True)
    d = d.sample(n=min(n_ctx, len(d)), random_state=SEED).reset_index(drop=True)
    return pd.DataFrame({"ctx_id": np.repeat(np.arange(len(d)), K),
                         "context": np.repeat(d["context"].values, K),
                         "cand": np.concatenate([list(e) for e in d["endings"]]),
                         "y": np.concatenate([(np.arange(K) == l).astype(int) for l in d["label"]])})

tok = lambda s: re.findall(r"\w+", s.lower())

def overlap(df):  # context-candidate overlap features, raw + centred within each context's 5 candidates
    ctoks = {i: tok(c) for i, c in df.drop_duplicates("ctx_id").set_index("ctx_id")["context"].items()}
    rows = []
    for i, c in zip(df.ctx_id, df.cand):
        ct, t = ctoks[i], tok(c); cs, ts = set(ct), set(t)
        cc, tc = cs - ENGLISH_STOP_WORDS, ts - ENGLISH_STOP_WORDS
        cb, tb = set(zip(ct, ct[1:])), set(zip(t, t[1:]))
        inter = len(cs & ts)
        rows.append([np.log1p(len(t)), inter / max(len(ts | cs), 1), inter / max(len(ts), 1),
                     len(cc & tc) / max(len(tc), 1), len(cb & tb) / max(len(tb), 1), len(cc & tc)])
    F = np.array(rows).reshape(-1, K, 6)
    return np.hstack([F.reshape(-1, 6), (F - F.mean(1, keepdims=True)).reshape(-1, 6)])

def top1(df, s): return float(np.mean(s.reshape(-1, K).argmax(1) == df.y.values.reshape(-1, K).argmax(1)))

def best_thr(y, s):  # F1-optimal threshold from out-of-fold train scores
    ts = np.quantile(s, np.linspace(0.5, 0.95, 46))
    return ts[int(np.argmax([f1_score(y, s >= t) for t in ts]))]

def run(model, grid, Xtr, Xte, tr, te, cv):
    gs = GridSearchCV(model, grid, scoring="average_precision", cv=cv, n_jobs=1).fit(Xtr, tr.y, groups=tr.ctx_id)
    sc = (lambda m, X: m.predict_proba(X)[:, 1])
    oof = cross_val_predict(gs.best_estimator_, Xtr, tr.y, cv=cv, groups=tr.ctx_id, method="predict_proba")[:, 1]
    thr = best_thr(tr.y.values, oof)  # threshold chosen on train CV only
    s = sc(gs.best_estimator_, Xte)
    return dict(f1=f1_score(te.y, s >= thr), top1=top1(te, s), auc=roc_auc_score(te.y, s)), gs.best_params_, thr

print("Loading data...")
tr, te = load_split("train", N_TRAIN_CTX), load_split("test", N_TEST_CTX)
print(f"train {len(tr)} rows, test {len(te)} rows")
if len(tr) < 5000 or len(te) < 1000: sys.exit("too little data")
cv = GroupKFold(n_splits=3)  # grouped by context so candidates of one context never straddle folds

print("Baseline: Naive Bayes on word 1-2gram counts")
nb = make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2), MultinomialNB())
b, bp, bt = run(nb, {"multinomialnb__alpha": [0.1, 0.5, 1.0]}, tr.cand, te.cand, tr, te, cv)
print(b, bp)

print("Improved: word+char TF-IDF + overlap features, class-weighted LR")
wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)
cvz = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100000)
wv.fit(tr.cand); cvz.fit(tr.cand)  # unsupervised vocabulary/IDF fit on train candidates only
sc = StandardScaler().fit(overlap(tr))

def build(df):
    cu = df.drop_duplicates("ctx_id")
    cos = np.asarray(wv.transform(df.cand).multiply(wv.transform(cu.context)[np.repeat(np.arange(len(cu)), K)]).sum(1))
    dense = np.hstack([sc.transform(overlap(df)), cos * 5])
    return sp.hstack([wv.transform(df.cand), cvz.transform(df.cand), sp.csr_matrix(dense)]).tocsr()

Xtr, Xte = build(tr), build(te)
lr = LogisticRegression(solver="liblinear", max_iter=200, random_state=SEED)
grid = {"C": [0.1, 0.3, 1, 3], "class_weight": [None, "balanced"]}
m, mp, mt = run(lr, grid, Xtr, Xte, tr, te, cv)
print(m, mp)

metrics = {"baseline_score": b["f1"], "improved_score": m["f1"], "improvement": m["f1"] - b["f1"],
           "baseline_f1": b["f1"], "improved_f1": m["f1"],
           "baseline_top1_acc": b["top1"], "improved_top1_acc": m["top1"],
           "baseline_roc_auc": b["auc"], "improved_roc_auc": m["auc"],
           "random_top1_acc": 1 / K}
if not all(np.isfinite(list(metrics.values()))): sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
names = ["F1 (true class)", "Top-1 acc", "ROC-AUC"]; keys = ["f1", "top1", "auc"]
x = np.arange(3)
plt.bar(x - .2, [b[k] for k in keys], .4, label="NB baseline"); plt.bar(x + .2, [m[k] for k in keys], .4, label="TF-IDF+overlap LR")
plt.xticks(x, names); plt.legend(); plt.title("CaseHOLD proxy, held-out test"); plt.tight_layout()
plt.savefig("figures/comparison.png", dpi=120)

res = {"question": "Can a tuned TF-IDF (word+char n-gram) model with context-candidate overlap features and class-imbalance handling beat a Naive Bayes n-gram baseline at recognizing genuine cited legal principles (holdings)?",
       "method": "CaseHOLD (LexGLUE) proxy: each context has 1 true holding and 4 distractors, each candidate a binary instance. Train contexts subsampled from the train split, held-out contexts subsampled from the official test split. Hyperparameters and decision threshold chosen by context-grouped 3-fold CV on train only; test used once.",
       "dataset": {"name": "LexGLUE CaseHOLD (proxy for BAILII corpus)", "source_url": "https://huggingface.co/datasets/coastalcph/lex_glue",
                   "n_rows": int(len(tr) + len(te)), "n_train": int(len(tr)), "n_test": int(len(te)), "same_as_paper": False},
       "metric_name": "F1 of the true-principle (positive) class", "higher_is_better": True,
       "baseline": "Multinomial Naive Bayes on word 1-2gram counts of the candidate (alpha tuned by CV).",
       "improved": "Logistic regression on word+char TF-IDF of the candidate plus context-candidate overlap features (raw and centred within context), CV-tuned C and class weighting.",
       "metrics": {k: float(v) for k, v in metrics.items()},
       "notes": "Proxy dataset (US case law, not UK BAILII) so results are not comparable to the paper. Both models use an F1-optimal threshold picked from out-of-fold train scores. TF-IDF vocabularies are fit unsupervised on train candidates before CV. Best params: baseline %s, improved %s. Single run, single seed, no confidence intervals; the test set is subsampled." % (bp, mp)}
with open("results.json", "w", encoding="utf-8") as f: json.dump(res, f, indent=2)
print("Done; wrote results.json")
