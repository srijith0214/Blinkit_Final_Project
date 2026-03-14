"""
rag_pipeline.py — Layer 4: Complete RAG Pipeline with Groq
===========================================================
Standalone module that implements the full RAG flow:

  Step 1 — RETRIEVAL
    • Load TF-IDF index (built by build_rag_index.py)
    • Vectorize user query using same TF-IDF vocabulary
    • Compute cosine similarity against all 5,000 feedback docs
    • Return top-K most relevant feedback chunks

  Step 2 — AUGMENTATION
    • Format retrieved chunks + business KPI context into a structured prompt
    • Inject into system message as grounded evidence

  Step 3 — GENERATION (Groq LLaMA 3)
    • Call Groq API (llama-3.3-70b-versatile or llama3-70b-8192)
    • Stream root-cause analysis back to caller
    • Supports multi-turn conversation history

Usage (standalone CLI):
    python rag_pipeline.py --query "Why are customers unhappy with delivery?" --api-key YOUR_GROQ_KEY

Usage (as module):
    from rag_pipeline import BlinkitRAG
    rag = BlinkitRAG(groq_api_key="gsk_...")
    answer = rag.query("Why is customer satisfaction low?")
    print(answer)
"""

import os
import pickle
import argparse
import json
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
IDX_PATH = os.path.join(BASE, "rag_index.pkl")

# ── Business KPI context (grounded in real dataset numbers) ──
BLINKIT_CONTEXT = """
BLINKIT BUSINESS INTELLIGENCE CONTEXT (from real dataset analysis):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 ORDERS (5,000 total | Mar 2023 – Nov 2024)
  • Total Revenue:     ₹1.10 Crore (₹11,009,308)
  • Avg Order Value:   ₹2,201.86
  • On-Time Rate:      69.4%  (3,470 on-time)
  • Slightly Delayed:  20.7%  (1,037 orders)
  • Significantly Delayed: 9.9% (493 orders)
  • Peak Delay Hours:  11 AM (37%), 1 PM (36.8%), 10 PM (37.4%)

📣 MARKETING (9 campaigns | 4 channels)
  • Total Ad Spend:    ₹1.63 Crore — EXCEEDS revenue
  • Overall ROAS:      0.67x  ← below 2.0x profitable threshold
  • Best Channel:      Email  2.05x ROAS  ✅ Only profitable channel
  • SMS ROAS:          1.99x
  • Social Media:      1.94x
  • App Notifications: 1.92x
  • Best Campaign:     Referral Program 2.03x

📦 TOP PRODUCT CATEGORIES (by revenue)
  1. Dairy & Breakfast    ₹6.39L
  2. Pharmacy             ₹5.92L
  3. Fruits & Vegetables  ₹5.59L
  4. Pet Care             ₹5.40L
  5. Household Care       ₹4.44L
  (Instant & Frozen Food: lowest at ₹3.07L)

👥 CUSTOMERS (2,500 total)
  • Regular:  639 (25.6%)
  • Premium:  633 (25.3%)
  • New:      628 (25.1%)
  • Inactive: 600 (24.0%)  ← High churn risk

⭐ CUSTOMER FEEDBACK (5,000 records)
  • Avg Rating:    3.34 / 5.0  (below 4.0 benchmark)
  • Positive:      1,620 (32.4%)
  • Negative:      1,642 (32.8%)
  • Neutral:       1,738 (34.8%)
  • Top complaint areas: Delivery (1,271), Customer Service (1,266),
    Product Quality (1,250), App Experience (1,213)
"""

SYSTEM_PROMPT_TEMPLATE = """You are an expert Business Intelligence Analyst for Blinkit, a quick commerce platform.

{blinkit_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED CUSTOMER FEEDBACK (top {k} semantically matched records via TF-IDF RAG):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{feedback_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTRUCTIONS:
Use BOTH the business KPI context above AND the retrieved customer feedback as evidence.
Structure your response as:
1. 🔍 Direct Answer — What is happening (cite specific feedback quotes as evidence)
2. 🧠 Root Cause Analysis — Why it is happening
3. 📊 Data Evidence — Reference specific numbers from the KPI context
4. ✅ Actionable Recommendations — 3-5 specific, prioritized steps management can take

Be concise, factual, and business-focused. Always quote specific feedback snippets as evidence.
"""

# ─────────────────────────────────────────────────────────────
class BlinkitRAG:
    """
    Full RAG pipeline: TF-IDF Retrieval → Context Augmentation → Groq Generation
    """
    load_dotenv()
    def __init__(self, groq_api_key: str = os.getenv("groq_api_key"), model: str = "llama-3.3-70b-versatile", top_k: int = 15):
        self.api_key = groq_api_key
        self.model   = model
        self.top_k   = top_k
        self.client  = Groq(api_key=groq_api_key)
        self._index  = None
        self.conversation_history: List[Dict] = []

        # Load index on init
        self._load_index()

    # ── Index loading ─────────────────────────────────────────
    def _load_index(self):
        if not os.path.exists(IDX_PATH):
            raise FileNotFoundError(
                f"RAG index not found at {IDX_PATH}\n"
                f"Run: python build_rag_index.py"
            )
        with open(IDX_PATH, "rb") as f:
            self._index = pickle.load(f)
        total = len(self._index["feedback_df"])
        features = self._index["tfidf_matrix"].shape[1]
        print(f"✅ RAG index loaded: {total} documents × {features} TF-IDF features")

    # ── Step 1: RETRIEVAL ─────────────────────────────────────
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Vectorize query using the same TF-IDF vocabulary,
        compute cosine similarity against all 5,000 feedback docs,
        return top-K most relevant records.
        """
        k = top_k or self.top_k
        vec = self._index["vectorizer"]
        mat = self._index["tfidf_matrix"]

        # Transform query into TF-IDF space
        query_vec = vec.transform([query])

        # Cosine similarity against all docs
        similarities = cosine_similarity(query_vec, mat).flatten()

        # Get top-K indices sorted by descending similarity
        top_indices = similarities.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            doc = self._index["feedback_df"][idx].copy()
            doc["similarity_score"] = round(float(similarities[idx]), 4)
            results.append(doc)
        return results

    # ── Step 2: AUGMENTATION ─────────────────────────────────
    def _build_context(self, retrieved: List[Dict]) -> str:
        """Format retrieved feedback chunks into a structured context block."""
        lines = []
        for i, r in enumerate(retrieved, 1):
            sentiment_icon = {"Positive": "✅", "Negative": "❌", "Neutral": "➖"}.get(
                r.get("sentiment", ""), "•"
            )
            lines.append(
                f"{i:>2}. {sentiment_icon} [{r['feedback_category']} | "
                f"Rating: {r['rating']}/5 | {r['sentiment']}] "
                f"(similarity: {r['similarity_score']:.3f})\n"
                f"    \"{r['feedback_text']}\""
            )
        return "\n".join(lines)

    def _build_system_prompt(self, retrieved: List[Dict]) -> str:
        """Build the full augmented system prompt with context injected."""
        feedback_ctx = self._build_context(retrieved)
        return SYSTEM_PROMPT_TEMPLATE.format(
            blinkit_context=BLINKIT_CONTEXT,
            k=len(retrieved),
            feedback_context=feedback_ctx
        )

    # ── Step 3: GENERATION (Groq) ─────────────────────────────
    def _call_groq(self, system_prompt: str, user_message: str) -> str:
        """Send augmented prompt to Groq LLaMA 3 and return response."""
        messages = [{"role": "system", "content": system_prompt}]

        # Include conversation history for multi-turn support
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,       # Low temp for factual business analysis
            top_p=0.9,
        )
        return response.choices[0].message.content

    # ── Full RAG Pipeline ─────────────────────────────────────
    def query(self, user_question: str, top_k: Optional[int] = None,
              remember: bool = True) -> Dict:
        """
        Execute full RAG pipeline:
          retrieve → augment → generate → return structured result

        Args:
            user_question: Natural language business question
            top_k: Number of feedback chunks to retrieve (default: self.top_k)
            remember: Whether to add this turn to conversation history

        Returns:
            dict with keys: answer, retrieved_chunks, system_prompt
        """
        # Step 1 — Retrieve
        retrieved = self.retrieve(user_question, top_k=top_k)

        # Step 2 — Augment
        system_prompt = self._build_system_prompt(retrieved)

        # Step 3 — Generate
        answer = self._call_groq(system_prompt, user_question)

        # Update conversation history for multi-turn
        if remember:
            self.conversation_history.append({"role": "user",    "content": user_question})
            self.conversation_history.append({"role": "assistant","content": answer})

        return {
            "answer":           answer,
            "retrieved_chunks": retrieved,
            "num_retrieved":    len(retrieved),
            "model":            self.model,
        }

    def reset_conversation(self):
        """Clear conversation history for a fresh session."""
        self.conversation_history = []
        print("🔄 Conversation history cleared.")

    def show_retrieved(self, chunks: List[Dict]):
        """Pretty-print retrieved feedback chunks."""
        print(f"\n{'─'*60}")
        print(f"  RETRIEVED FEEDBACK CHUNKS ({len(chunks)} results)")
        print(f"{'─'*60}")
        for i, c in enumerate(chunks, 1):
            icon = {"Positive":"✅","Negative":"❌","Neutral":"➖"}.get(c.get("sentiment",""),"•")
            print(f"\n  {i}. {icon} [{c['feedback_category']}] Rating: {c['rating']}/5 "
                  f"| Similarity: {c['similarity_score']:.3f}")
            print(f"     \"{c['feedback_text']}\"")


# ─────────────────────────────────────────────────────────────
# CLI — run standalone
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Blinkit RAG Pipeline — TF-IDF Retrieval + Groq LLaMA 3"
    )
    parser.add_argument("--query",   "-q", required=True,  help="Business question to analyze")
    parser.add_argument("--api-key", "-k", required=True,  help="Groq API key")
    parser.add_argument("--top-k",   "-n", type=int, default=15, help="Feedback chunks to retrieve (default: 15)")
    parser.add_argument("--model",   "-m", default="llama-3.3-70b-versatile",
                        choices=["llama-3.3-70b-versatile","llama3-70b-8192","mixtral-8x7b-32768"],
                        help="Groq model to use")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved feedback chunks")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive multi-turn chat")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  Blinkit RAG Pipeline — Groq LLaMA 3")
    print("═"*60)
    print(f"  Model:   {args.model}")
    print(f"  Top-K:   {args.top_k} feedback chunks")
    print("═"*60 + "\n")

    rag = BlinkitRAG(
        groq_api_key=args.api_key,
        model=args.model,
        top_k=args.top_k
    )

    if args.interactive:
        # Multi-turn interactive mode
        print("🤖 Interactive RAG Chat (type 'quit' to exit, 'reset' to clear history)\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not question:
                continue
            if question.lower() in ("quit","exit","q"):
                break
            if question.lower() == "reset":
                rag.reset_conversation()
                continue

            result = rag.query(question)
            if args.show_chunks:
                rag.show_retrieved(result["retrieved_chunks"])
            print(f"\n🤖 AI Analysis:\n{result['answer']}\n")
            print("─"*60)
    else:
        # Single query mode
        print(f"❓ Query: {args.query}\n")
        result = rag.query(args.query)

        if args.show_chunks:
            rag.show_retrieved(result["retrieved_chunks"])

        print(f"\n{'═'*60}")
        print("  GROQ LLaMA 3 — ROOT CAUSE ANALYSIS")
        print(f"{'═'*60}\n")
        print(result["answer"])
        print(f"\n{'─'*60}")
        print(f"Retrieved: {result['num_retrieved']} feedback chunks | Model: {result['model']}")


if __name__ == "__main__":
    main()
