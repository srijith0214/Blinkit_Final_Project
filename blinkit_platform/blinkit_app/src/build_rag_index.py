"""
build_rag_index.py — Layer 4: RAG Index Builder
================================================
Builds a TF-IDF retrieval index over 5,000 customer feedback rows.
Uses bigram TF-IDF vectors for semantic similarity search.
Saved as src/rag_index.pkl

Pipeline:
  1. Load + enrich feedback with order metadata
  2. Build TF-IDF vectorizer (unigrams + bigrams, 5000 features)
  3. Save vectorizer + matrix + raw records for retrieval
  4. At query time: transform query → cosine similarity → top-K results → Groq LLM

Usage: python build_rag_index.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), "data")
OUT_PATH = os.path.join(BASE, "rag_index.pkl")

print("=" * 60)
print("Blinkit RAG Index Builder — TF-IDF on Customer Feedback")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────
feedback = pd.read_csv(os.path.join(DATA_DIR, "blinkit_customer_feedback.csv"))
orders   = pd.read_csv(os.path.join(DATA_DIR, "blinkit_orders.csv"))

print(f"Feedback rows: {len(feedback)}")

# ── Enrich feedback with order context ─────────────────────────
fb = feedback.merge(
    orders[['order_id','delivery_status','order_total','payment_method']],
    on='order_id', how='left'
)
fb['feedback_text'] = fb['feedback_text'].fillna('')

# Enriched document for TF-IDF (category + sentiment + text)
fb['enriched'] = fb.apply(lambda r:
    f"{r['feedback_category']} {r['sentiment']} rating {r['rating']} "
    f"{r.get('delivery_status','')} {r['feedback_text']}", axis=1
)

# ── Build TF-IDF index ─────────────────────────────────────────
print("Building TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words='english',
    min_df=2,
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(fb['enriched'])
print(f"TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

# ── Validate retrieval with a test query ───────────────────────
test_query = "damaged product poor quality"
q_vec = vectorizer.transform([test_query])
sims  = cosine_similarity(q_vec, tfidf_matrix).flatten()
top5  = sims.argsort()[-5:][::-1]
print(f"\nTest retrieval: '{test_query}'")
for i in top5:
    print(f"  [{fb.iloc[i]['sentiment']}|{fb.iloc[i]['feedback_category']}] "
          f"{fb.iloc[i]['feedback_text'][:70]}...")

# ── Save payload ───────────────────────────────────────────────
payload = {
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'feedback_df': fb[[
        'feedback_id','order_id','customer_id','rating',
        'feedback_text','feedback_category','sentiment',
        'feedback_date','delivery_status','order_total'
    ]].to_dict('records')
}
with open(OUT_PATH, 'wb') as f:
    pickle.dump(payload, f)

print(f"\n✅ RAG index saved to {OUT_PATH}")
print(f"   Use retrieve_feedback(query, payload, top_k=15) to query at runtime")
