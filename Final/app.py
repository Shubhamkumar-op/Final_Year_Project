import warnings
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_client_cls" in .* has conflict with protected namespace "model_".*',
    category=UserWarning,
)

import streamlit as st
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


def main():
    st.set_page_config(page_title="📚 Divyansh LLM", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        /* Main container */
        .stApp {
            background-color: #1e1e2f;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title */
        h1 {
            color: #ffffff;
            text-shadow: 1px 1px 2px #000;
        }

        /* Sidebar */
        .css-1d391kg {
            background-color: #262730 !important;
            color: white;
            border-right: 2px solid #444;
        }

        /* Text input */
        .stTextInput>div>div>input {
            background-color: #333 !important;
            color: white !important;
            border-radius: 5px;
        }

        /* Buttons */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 16px;
        }

        /* Markdown and text */
        .stMarkdown, .stDataFrame, .stText, .stSubheader {
            font-size: 16px;
            line-height: 1.6;
        }

        /* Chunk card */
        .chunk-card {
            background-color: #202235;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            color: #e0e0e0;
        }

        .chunk-title {
            color: #f9c74f;
            font-weight: 600;
        }

        /* Final answer */
        .final-answer {
            background-color: #1b263b;
            border-left: 5px solid #4CAF50;
            padding: 16px;
            border-radius: 6px;
            color: white;
            font-size: 17px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📚 Divyansh LLM")
    st.write("Ask questions based on the uploaded PDFs.")

    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(storage_dir, uploaded_file.name)
            if not os.path.exists(pdf_path):
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.sidebar.success(f"Saved {uploaded_file.name}")
                # Process embedding for new PDF
                store_pdf_embedding(pdf_path)
            else:
                st.sidebar.info(f"{uploaded_file.name} already processed")

    # Show loaded PDFs
    pdf_files = [f for f in os.listdir(storage_dir) if f.endswith(".pdf")]
    st.sidebar.markdown(f"**Loaded PDFs:** {', '.join(pdf_files) if pdf_files else 'None'}")

    # Initialize QueryAgent once in session state
    if "agent" not in st.session_state:
        st.session_state.agent = QueryAgent()

    # User question input
    user_query = st.text_input("📝 Your Question:")

    if user_query:
        with st.spinner("🔎 Searching for relevant info..."):
            context = st.session_state.agent.search_all(user_query)

        if not context:
            st.warning("⚠️ No relevant info found in the PDFs.")
        else:
            st.subheader("📄 Retrieved PDF Chunks:")
            for pdf, page, chunk, dist in context:
                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <div class="chunk-title">📘 {pdf} — Page {page}</div>
                        <div>{chunk[:300]}...</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.spinner("💬 Generating answer..."):
                answer = generate_final_answer(user_query, context)
            st.subheader("💡 Final Answer:")
            st.markdown(
                f"""
                <div class="final-answer">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
