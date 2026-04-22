from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

# Embedding (lightweight)
embedding = FakeEmbeddings(size=384)

# Load DB
db = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

# Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",   
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Intent detection
def detect_intent(query):
    q = query.lower()
    if "refund" in q:
        return "policy"
    elif "delivery" in q or "shipping" in q:
        return "shipping"
    elif "cancel" in q:
        return "cancellation"
    elif "hello" in q:
        return "greeting"
    return "unknown"

# Confidence
def calculate_confidence(docs):
    return 0.8 if len(docs) > 0 else 0.0


# MAIN FUNCTION
def process_query(query):
    # 🔥 Direct similarity search (stable)
    docs = db.similarity_search(query, k=5)

    print("DEBUG → Docs found:", len(docs))

    context = "\n".join([d.page_content for d in docs])

    if not context:
        return "No relevant info found", 0.0, "unknown"

    answer = llm.invoke(
        f"""
You are a customer support assistant.

Answer ONLY from the context below.

If the answer is not present, say:
"I don't know based on the provided document."

Context:
{context}

Question: {query}
"""
    ).content

    confidence = calculate_confidence(docs)
    intent = detect_intent(query)

    return answer, confidence, intent