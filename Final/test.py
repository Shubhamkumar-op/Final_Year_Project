import warnings
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_client_cls" in .* has conflict with protected namespace "model_".*',
    category=UserWarning,
)

import os
import fitz  # PyMuPDF
import faiss
import pickle
import textwrap
from sentence_transformers import SentenceTransformer
from autogen import ConversableAgent

# LLM Config: Mistral via Ollama
llm_config = {
    "model": "mistral:latest",
    "base_url": "http://localhost:11434",
    "api_type": "ollama"
}

# Directory to store FAISS index and chunked text
storage_dir = "pdf_data"
os.makedirs(storage_dir, exist_ok=True)


class PDFAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text_by_page(self):
        try:
            doc = fitz.open(self.pdf_path)
            pages = [(page_num + 1, page.get_text()) for page_num, page in enumerate(doc)]
            return pages
        except Exception as e:
            print(f"[!] Failed to read PDF {self.pdf_path}: {e}")
            return []


class EmbeddingAgent:
    def __init__(self, pdf_name):
        try:
            self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print("[!] Failed to load embedding model:", e)
        self.pdf_name = pdf_name

    def chunk_text(self, pages, chunk_size=200):
        chunks = []
        for page_num, text in pages:
            wrapped = textwrap.wrap(text, width=chunk_size)
            for chunk in wrapped:
                chunks.append((page_num, chunk))
        return chunks

    def build_index(self, chunks):
        texts = [chunk for _, chunk in chunks]
        embeddings = self.emb_model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, texts

    def store(self, index, chunks):
        base = os.path.join(storage_dir, self.pdf_name)
        faiss.write_index(index, f"{base}.index")
        with open(f"{base}.pkl", "wb") as f:
            pickle.dump(chunks, f)


def store_pdf_embedding(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base = os.path.join(storage_dir, pdf_name)

    if os.path.exists(f"{base}.index") and os.path.exists(f"{base}.pkl"):
        print(f"[✓] Already exists: {pdf_name}")
        return

    print(f"[*] Processing {pdf_name}...")
    text = PDFAgent(pdf_path).extract_text_by_page()
    if not text:
        return

    embedding_agent = EmbeddingAgent(pdf_name)
    chunks = embedding_agent.chunk_text(text)
    index, _ = embedding_agent.build_index(chunks)
    embedding_agent.store(index, chunks)
    print(f"[✓] Stored: {pdf_name}")


class QueryAgent:
    def __init__(self):
        try:
            self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print("[!] Failed to load embedding model:", e)

    def search_all(self, query, top_k=3):
        query_vec = self.emb_model.encode([query])
        results = []
        for file in os.listdir(storage_dir):
            if file.endswith(".index"):
                pdf_name = file.replace(".index", "")
                try:
                    index = faiss.read_index(os.path.join(storage_dir, file))
                    with open(os.path.join(storage_dir, f"{pdf_name}.pkl"), "rb") as f:
                        chunks = pickle.load(f)
                    D, I = index.search(query_vec, top_k)
                    for dist, idx in zip(D[0], I[0]):
                        if idx < len(chunks):
                            page_num, chunk = chunks[idx]
                            results.append((pdf_name, page_num, chunk, dist))
                except Exception as e:
                    print(f"[!] Error reading {pdf_name}: {e}")
        results.sort(key=lambda x: x[3])
        return results[:top_k]


def generate_final_answer(query, retrieved_contexts):
    llm_agent = ConversableAgent(
        name="AnswerAgent",
        system_message=(
            "You are a knowledgeable assistant. First, answer the question using only the PDF chunks provided, "
            "mentioning the PDF names and page numbers. Then, based on your general knowledge, expand the answer "
            "with any additional insights not found in the PDFs."
        ),
        llm_config=llm_config
    )

    print("\n[Context Preview]")
    for pdf, page, chunk, _ in retrieved_contexts:
        print(f"- PDF: {pdf}, Page: {page}\n  → {chunk[:150]}...\n")

    context_str = "\n\n".join([
        f"From PDF '{pdf}' (Page {page}):\n{chunk}" for pdf, page, chunk, _ in retrieved_contexts
    ])

    prompt = (
        f"The following are excerpts from different PDFs:\n\n"
        f"{context_str}\n\n"
        f"User question: {query}\n\n"
        "Please do the following:\n"
        "1. Answer the question using only the content from the PDFs above.\n"
        "2. Then, extend your answer by including any general knowledge, real-world information, or technical details you know, "
        "even if not found in the PDFs.\n\n"
        "Clearly label the two parts of your answer."
    )

    reply = llm_agent.generate_reply(messages=[{"role": "user", "name": "UserInput", "content": prompt}])
    return reply.get("content", "No response.")


if __name__ == "__main__":
    pdf_list = ["genai-principles.pdf", "LLM.pdf", "researchpaper4.pdf"]
    for pdf_file in pdf_list:
        store_pdf_embedding(pdf_file)

    agent = QueryAgent()

    print("\n📚 PDF QA System Ready! Ask questions based on the uploaded PDFs.")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("📝 Your Question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Have a great day!")
            break

        context = agent.search_all(user_query)
        if not context:
            print("⚠️ No relevant context found in the PDFs.")
            continue

        answer = generate_final_answer(user_query, context)
        print("\n💡 Answer:\n", answer)
