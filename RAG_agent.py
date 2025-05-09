import warnings

warnings.filterwarnings(
    "ignore",
    message=r'Field "model_client_cls" in .* has conflict with protected namespace "model_".*',
    category=UserWarning,
)

import os
import fitz  
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from autogen import ConversableAgent

llm_config = {
    "model": "gemma:2b",
    "base_url": "http://localhost:11434",
    "api_type": "ollama"
}

class PDFExtractor:
    def __init__(self, pdf_path, faiss_index_path="index.faiss", data_path="data.pkl"):
        self.pdf_path = pdf_path
        self.faiss_index_path = faiss_index_path
        self.data_path = data_path
        self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text

    def chunk_text(self, text, chunk_size=200):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def build_faiss_index(self):
        text = self.extract_text()
        self.text_chunks = self.chunk_text(text)
        embeddings = self.emb_model.encode(self.text_chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, self.faiss_index_path)
        with open(self.data_path, "wb") as f:
            pickle.dump(self.text_chunks, f)

class PDFSearchAgent:
    def __init__(self, faiss_index_path="index.faiss", data_path="data.pkl"):
        self.faiss_index_path = faiss_index_path
        self.data_path = data_path
        self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.data_path, "rb") as f:
            self.text_chunks = pickle.load(f)

    def search(self, query, top_k=3):
        query_vec = self.emb_model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        results = [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]
        return " ".join(results) if results else "NOT_FOUND"


user_query = "What is General-Purpose Language Models"

pdf_path = "genai-principles.pdf"
extractor = PDFExtractor(pdf_path)
extractor.build_faiss_index()


search_agent = PDFSearchAgent()
search_result = search_agent.search(user_query)


llm_agent = ConversableAgent(
    name="ExternalKnowledgeAgent",
    system_message="You answer queries using external knowledge if the document does not contain an answer.",
    llm_config=llm_config
)


context = (
    f"Context from resume:\n{search_result}\n\n"
    f"User question: {user_query}\n\n"
    "Based on the above context, give a detailed and smart response. If needed, use your own knowledge to elaborate."
)

final_response = llm_agent.generate_reply(
    messages=[{"role": "user", "name": "UserInputHandler", "content": context}]
).get("content", "")


print("\n Final Answer ")
print(final_response)
