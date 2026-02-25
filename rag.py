import os
import re
import time
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()  # must be before os.getenv calls

from google import genai
from flask import Flask, request, jsonify
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────────────

MONGO_URI      = os.getenv("MONGO_URL")
DB_NAME        = os.getenv("DB_NAME",      "pfe_project")
COLLECTION     = os.getenv("COLLECTION",   "tunisian_law")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PORT           = int(os.getenv("PORT",     6000))
TOP_K          = int(os.getenv("TOP_K",    5))

if not MONGO_URI:
    raise RuntimeError("MONGO_URL is not set in .env")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── CORS with Private Network Access support ─────────────────────────────────
@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Private-Network'] = 'true'
        return response

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    return response

# ── STARTUP: load everything once at boot ─────────────────────────────────────

print("=" * 60)
print("Arabic Legal RAG — Flask Microservice (Multilingual)")
print("=" * 60)

print("\nLoading embedding model (multilingual-e5-large)...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
print("  Model ready.")

print("Connecting to MongoDB...")
mongo_client = MongoClient(MONGO_URI)
collection   = mongo_client[DB_NAME][COLLECTION]
doc_count    = collection.count_documents({})
print(f"  Connected — {doc_count} legal articles available.")

print("Initializing Gemini...")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
ACTIVE_GEMINI_MODEL = GEMINI_MODEL
print(f"  Gemini ready ({ACTIVE_GEMINI_MODEL}).")

print(f"\nServer running on http://localhost:{PORT}")
print("=" * 60)


# ── LANGUAGE DETECTION ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect whether the question is Arabic, French, or English.
    Returns: 'arabic' | 'french' | 'english'
    """
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    arabic_ratio = arabic_chars / max(len(text.strip()), 1)

    if arabic_ratio > 0.2:
        return 'arabic'

    french_words   = r'\b(le|la|les|de|du|des|est|sont|pour|avec|dans|sur|une|un|comment|quoi|quel|quelle|qui|que|quand|où|quelles|quels|puis-je|puis|pouvez|voulez)\b'
    french_accents = r'[àâäéèêëîïôùûüç]'
    french_score   = len(re.findall(french_words, text.lower())) + \
                     len(re.findall(french_accents, text.lower()))

    if french_score >= 1:
        return 'french'

    return 'english'


# ── PROMPTS ───────────────────────────────────────────────────────────────────

def build_context(articles: list) -> str:
    """Build the legal context block shared across all prompts."""
    context = ""
    for i, r in enumerate(articles):
        context += (
            f"\n[Source {i+1}] "
            f"Article {r['article_num']} (Law {r['law_num']}) "
            f"— relevance: {r['score']:.2f}\n"
            f"{r['text']}\n"
            f"{'─' * 40}"
        )
    return context


def build_unified_prompt(question: str, articles: list, lang: str) -> str:
    """
    Single unified prompt that combines DB context (if any) with a web search
    instruction. Gemini is asked to:
      1. Use the retrieved DB articles as a primary grounding source.
      2. Use its Google Search tool to complement / fill gaps.
      3. Synthesise one coherent answer from both sources.

    `articles` may be an empty list when no DB results exist — the prompt
    degrades gracefully to web-search-only in that case.
    """
    has_db_context = bool(articles)
    context        = build_context(articles) if has_db_context else ""

    if lang == 'arabic':
        db_section = f"""
=== النصوص المستردة من قاعدة البيانات ===
{context}
""" if has_db_context else """
=== قاعدة البيانات ===
لم يتم العثور على نصوص ذات صلة كافية في قاعدة البيانات المحلية.
"""
        return f"""أنت مساعد قانوني متخصص في القانون التونسي ولديك أداة بحث على الإنترنت.

مهمتك: تقديم إجابة قانونية شاملة على سؤال المستخدم من خلال **مزج مصدرين**:
1. النصوص المستردة من قاعدة البيانات المحلية (إن وجدت)
2. بحث إضافي على الإنترنت عن مصادر القانون التونسي

خطوات العمل:
- ابدأ بمراجعة النصوص الواردة من قاعدة البيانات
- ابحث على الإنترنت لتكملة أي نقص أو تأكيد المعلومات (مجلة الالتزامات والعقود، مجلة الأحوال الشخصية، مجلة الشغل، المجلة التجارية، القانون الجزائي، الدستور، إلخ)
- اجمع المصدرين في إجابة واحدة متماسكة

قواعد الإجابة:
- اذكر رقم الفصل دائماً عند الاستناد (مثال: "وفقاً للفصل 16...")
- ميّز بين ما جاء من قاعدة البيانات وما وجدته عبر البحث الإلكتروني
- أجب بالعربية دائماً
- أنهِ بتوصية بالتحقق من المصادر الرسمية أو استشارة محامٍ
{db_section}
=== سؤال المستخدم ===
{question}

=== الإجابة القانونية الشاملة ==="""

    elif lang == 'french':
        db_section = f"""
=== Textes récupérés depuis la base de données ===
{context}
""" if has_db_context else """
=== Base de données ===
Aucun texte suffisamment pertinent n'a été trouvé dans la base de données locale.
"""
        return f"""Tu es un assistant juridique spécialisé en droit tunisien disposant d'un outil de recherche web.

Ta mission : fournir une réponse juridique complète en **combinant deux sources** :
1. Les textes récupérés depuis la base de données locale (si disponibles)
2. Une recherche complémentaire sur Internet pour trouver des sources juridiques tunisiennes

Étapes de travail :
- Commence par examiner les textes fournis par la base de données
- Effectue une recherche web pour combler les lacunes ou confirmer les informations (COC, Code du Statut Personnel, Code du Travail, Code de Commerce, Code Pénal, Constitution, etc.)
- Synthétise les deux sources en une réponse unique et cohérente

Règles de réponse :
- Cite toujours le numéro d'article lors de chaque référence (ex. : "Selon l'article 16...")
- Distingue ce qui provient de la base de données de ce que tu as trouvé via la recherche web
- Réponds toujours en français
- Termine en recommandant de vérifier les sources officielles ou de consulter un avocat
{db_section}
=== Question de l'utilisateur ===
{question}

=== Réponse juridique complète ==="""

    else:  # english
        db_section = f"""
=== Texts retrieved from the database ===
{context}
""" if has_db_context else """
=== Database ===
No sufficiently relevant texts were found in the local database.
"""
        return f"""You are a legal assistant specialized in Tunisian law with access to a web search tool.

Your mission: provide a comprehensive legal answer by **combining two sources**:
1. Texts retrieved from the local database (if available)
2. Additional web search to find Tunisian legal sources online

Steps:
- Start by reviewing the database texts provided
- Search the web to fill any gaps or confirm information (COC, Personal Status Code, Labour Code, Commercial Code, Penal Code, Constitution, decrees, etc.)
- Merge both sources into one single coherent answer

Answer rules:
- Always cite the article number when referencing it (e.g. "According to Article 16...")
- Distinguish between what came from the database and what you found via web search
- Always respond in English
- End by recommending verification with official sources or consultation with a lawyer
{db_section}
=== User's question ===
{question}

=== Comprehensive legal answer ==="""


def classify_gemini_error(error_text: str) -> str:
    """Classify Gemini API errors for cleaner API responses."""
    text = error_text.lower()
    if "resource_exhausted" in text or "quota" in text or "429" in text:
        return "quota_exceeded"
    if "invalid_argument" in text or "400" in text:
        return "invalid_request"
    if "permission_denied" in text or "403" in text:
        return "permission_denied"
    if "unauthorized" in text or "401" in text:
        return "unauthorized"
    if "not_found" in text or "404" in text:
        return "model_not_found"
    return "unknown"


def generate_with_gemini(prompt: str) -> tuple[str, str]:
    """Generate content with configured model and graceful fallback if unavailable."""
    global ACTIVE_GEMINI_MODEL

    preferred_candidates = [
        ACTIVE_GEMINI_MODEL,
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    model_candidates = []
    seen_models = set()
    for model_name in preferred_candidates:
        if model_name and model_name not in seen_models:
            model_candidates.append(model_name)
            seen_models.add(model_name)

    errors = []

    for model_name in model_candidates:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            ACTIVE_GEMINI_MODEL = model_name
            return (response.text or "", model_name)
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            continue

    raise RuntimeError(
        "Gemini generation failed for all model candidates. "
        + " | ".join(errors)
    )


def generate_with_gemini_search(prompt: str) -> tuple[str, str]:
    """
    Generate content using Gemini with Google Search grounding enabled.
    Used for the fallback path when no relevant DB articles are found.
    Tries search-capable models first, then falls back to standard generation.
    """
    global ACTIVE_GEMINI_MODEL

    # Models that support Google Search grounding
    search_candidates = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        ACTIVE_GEMINI_MODEL,
    ]

    model_candidates = []
    seen_models = set()
    for model_name in search_candidates:
        if model_name and model_name not in seen_models:
            model_candidates.append(model_name)
            seen_models.add(model_name)

    errors = []

    # ── Attempt 1: Gemini with Google Search grounding ────────────────────────
    for model_name in model_candidates:
        try:
            from google.genai import types as genai_types

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                ),
            )
            ACTIVE_GEMINI_MODEL = model_name

            # Extract plain text; grounded responses may include citation metadata
            text = response.text or ""
            return (text, model_name)

        except Exception as exc:
            errors.append(f"{model_name}+search: {exc}")
            continue

    # ── Attempt 2: Plain generation without search (last resort) ─────────────
    try:
        plain_text, plain_model = generate_with_gemini(prompt)
        return (plain_text, plain_model)
    except Exception as exc:
        errors.append(f"plain_fallback: {exc}")

    raise RuntimeError(
        "Gemini search generation failed for all candidates. "
        + " | ".join(errors)
    )


# ── HELPERS ───────────────────────────────────────────────────────────────────

def vector_search(query: str, top_k: int = TOP_K, law_num: int = None) -> list:
    """Embed query and run MongoDB Atlas $vectorSearch."""
    embedding = embed_model.encode(query).tolist()

    vector_stage = {
        "index":         "embedding_index",
        "path":          "embedding",
        "queryVector":   embedding,
        "numCandidates": top_k * 10,
        "limit":         top_k,
    }

    if law_num is not None:
        vector_stage["filter"] = {"law_num": {"$eq": int(law_num)}}

    pipeline = [
        {"$vectorSearch": vector_stage},
        {
            "$project": {
                "_id":         0,
                "chunk_id":    1,
                "law_num":     1,
                "article_num": 1,
                "page":        1,
                "text":        1,
                "score":       {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))




# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — call this from NestJS to verify the service is up."""
    return jsonify({
        "status":           "ok",
        "model":            "intfloat/multilingual-e5-large",
        "gemini":           ACTIVE_GEMINI_MODEL,
        "articles":         doc_count,
        "db":               DB_NAME,
        "collection":       COLLECTION,
        "languages":        ["arabic", "french", "english"],
        "answerStrategy":   "rag+web_search (always combined)",
    })


@app.post("/search")
def search():
    """
    Vector search only — no Gemini. Useful for debugging.

    Body: { "question": "...", "topK": 5, "lawNum": null }
    """
    body     = request.get_json() or {}
    question = (body.get("question") or "").strip()
    top_k    = int(body.get("topK", TOP_K))
    law_num  = body.get("lawNum")

    if not question:
        return jsonify({"error": "question is required"}), 400

    start   = time.time()
    lang    = detect_language(question)
    results = vector_search(question, top_k, law_num)
    elapsed = int((time.time() - start) * 1000)

    return jsonify({
        "question":         question,
        "language":         lang,
        "results":          results,
        "count":            len(results),
        "processingTimeMs": elapsed,
    })


@app.post("/ask")
def ask():
    """
    Full RAG pipeline — always combines DB context AND Gemini web search.

    Flow:
      1. Detect language
      2. Vector search MongoDB (all results returned regardless of score)
      3. Build unified prompt with DB context (or empty section if none found)
      4. Gemini generates answer using BOTH the DB context AND live web search
         → one coherent answer synthesised from both sources

    Body:
      {
        "question": "ما هي شروط الوعد بالجعل؟",
        "topK":     5,   // optional, default from .env
        "lawNum":   1    // optional, filter to a specific law (1–5)
      }

    Response:
      {
        "question":         "...",
        "language":         "arabic",
        "answer":           "وفقاً للفصل 20... (قاعدة البيانات) ... وبحسب ما وجدناه على الإنترنت...",
        "answerSource":     "rag+web_search",
        "sources":          [ { "chunk_id": "...", "article_num": 20, ... } ],
        "processingTimeMs": 1243
      }
    """
    body     = request.get_json() or {}
    question = (body.get("question") or "").strip()
    top_k    = int(body.get("topK", TOP_K))
    law_num  = body.get("lawNum")

    if not question:
        return jsonify({"error": "question is required"}), 400

    start = time.time()

    # 1. Detect language
    lang = detect_language(question)

    # 2. Vector search MongoDB (best-effort; may return empty or low-score results)
    articles = vector_search(question, top_k, law_num)

    # 3. Build unified prompt (DB context + web search instruction)
    prompt = build_unified_prompt(question, articles, lang)

    # 4. Generate with Gemini + Google Search grounding
    try:
        answer, used_model = generate_with_gemini_search(prompt)
    except Exception as exc:
        details = str(exc)
        return jsonify({
            "error":   "Gemini request failed",
            "reason":  classify_gemini_error(details),
            "details": details,
            "hint":    "Check Gemini API key quota/billing and allowed models.",
        }), 502

    return jsonify({
        "question":         question,
        "language":         lang,
        "answer":           answer,
        "answerSource":     "rag+web_search",
        "geminiModel":      used_model,
        "sources":          articles,
        "processingTimeMs": int((time.time() - start) * 1000),
    })


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)