import warnings
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_client_cls" in .* has conflict with protected namespace "model_".*',
    category=UserWarning,
)

import streamlit as st
import os
import fitz
import faiss
import pickle
import textwrap
from sentence_transformers import SentenceTransformer
from autogen import ConversableAgent
import argostranslate.translate

storage_dir = "pdf_data"
os.makedirs(storage_dir, exist_ok=True)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_translation_languages():
    return argostranslate.translate.get_installed_languages()

@st.cache_data
def load_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return [(page_num + 1, page.get_text()) for page_num, page in enumerate(doc)]
    except Exception as e:
        st.error(f"Failed to read PDF {pdf_path}: {e}")
        return []

def offline_translate_to_hindi(text: str) -> str:
    installed_languages = load_translation_languages()
    from_lang = next(filter(lambda x: x.code == "en", installed_languages), None)
    to_lang = next(filter(lambda x: x.code == "hi", installed_languages), None)
    if from_lang is None or to_lang is None:
        return "Hindi translation model not installed."
    translation = from_lang.get_translation(to_lang)
    return translation.translate(text)

llm_config = {
    "model": "mistral:latest",
    "base_url": "http://localhost:11434",
    "api_type": "ollama"
}

class EmbeddingAgent:
    def __init__(self, pdf_name):
        self.emb_model = load_embedding_model()
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
        st.info(f"Embeddings for '{pdf_name}' already exist. Skipping processing.")
        return

    st.info(f"Processing {pdf_name} embeddings...")
    text = load_pdf_text(pdf_path)
    if not text:
        return

    embedding_agent = EmbeddingAgent(pdf_name)
    chunks = embedding_agent.chunk_text(text)
    index, _ = embedding_agent.build_index(chunks)
    embedding_agent.store(index, chunks)
    st.success(f"Stored embeddings for '{pdf_name}'.")

class QueryAgent:
    def __init__(self):
        self.emb_model = load_embedding_model()
        self.indices = {}
        self.chunks = {}
        self._load_all_embeddings()

    def _load_all_embeddings(self):
        for file in os.listdir(storage_dir):
            if file.endswith(".index"):
                pdf_name = file.replace(".index", "")
                try:
                    index_path = os.path.join(storage_dir, file)
                    pkl_path = os.path.join(storage_dir, f"{pdf_name}.pkl")
                    if pdf_name not in self.indices:
                        index = faiss.read_index(index_path)
                        with open(pkl_path, "rb") as f:
                            chunks = pickle.load(f)
                        self.indices[pdf_name] = index
                        self.chunks[pdf_name] = chunks
                except Exception as e:
                    st.error(f"Error loading embeddings for '{pdf_name}': {e}")

    def search_all(self, query, top_k=3):
        query_vec = self.emb_model.encode([query])
        results = []
        for pdf_name, index in self.indices.items():
            chunks = self.chunks.get(pdf_name, [])
            D, I = index.search(query_vec, top_k)
            for dist, idx in zip(D[0], I[0]):
                if idx < len(chunks):
                    page_num, chunk = chunks[idx]
                    results.append((pdf_name, page_num, chunk, dist))
        results.sort(key=lambda x: x[3])  # sort by distance
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
    english_answer = reply.get("content", "No response.")
    return english_answer

def main():
    st.set_page_config(page_title="LLM PDF Q&A with Hindi Translation", layout="wide")
    st.title("📚 LLM PDF Q&A with Hindi Translation")

    stored_pdfs = [f for f in os.listdir(storage_dir) if f.endswith(".pkl")]
    st.info(f"📂 Total PDFs in database: {len(stored_pdfs)}")

    if stored_pdfs:
        pdf_names = [os.path.splitext(name)[0] for name in stored_pdfs]
        st.markdown("### 📄 Stored PDF files:")
        for name in pdf_names:
            st.markdown(f"- {name}")

    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(storage_dir, uploaded_file.name)
            if not os.path.exists(pdf_path):
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name}")
            store_pdf_embedding(pdf_path)

    query = st.text_input("Ask a question about the PDFs content:")
    if query:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            qa_agent = QueryAgent()
            results = qa_agent.search_all(query)

            if results:
                st.subheader("Relevant PDF chunks:")
                for pdf_name, page_num, chunk, dist in results:
                    st.markdown(f"**From {pdf_name} (Page {page_num}):**")
                    st.write(chunk)

                english_answer = generate_final_answer(query, results)

                st.subheader("Answer (English):")
                st.markdown(
                    f"<div style='background-color:#1b263b; border-left:5px solid #4CAF50; padding:10px; margin-top:20px; color:white;'>{english_answer}</div>",
                    unsafe_allow_html=True,
                )

                if st.button("🔁 Translate to Hindi"):
                    with st.spinner("Translating to Hindi..."):
                        hindi_answer = offline_translate_to_hindi(english_answer)
                        st.subheader("Answer (Hindi - Offline Translation):")
                        st.markdown(
                            f"<div style='background-color:#1b263b; border-left:5px solid #4CAF50; padding:10px; margin-top:20px; color:white;'>{hindi_answer}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("No relevant content found in uploaded PDFs.")
    else:
        st.info("Please upload one or more PDF files and ask a question to get started.")

if __name__ == "__main__":
    main()
