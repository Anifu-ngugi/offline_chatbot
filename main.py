import os
import tempfile
import chromadb
import ollama
import pysqlite3 as sqlite3
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from concurrent.futures import ThreadPoolExecutor

# System prompt for the LLM
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

@st.cache_data
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks."""
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    os.unlink(temp_file_path)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

@st.cache_resource
def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage."""
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma2")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str, batch_size: int = 10):
    """Adds document splits to a vector collection in batches."""
    collection = get_vector_collection()

    def process_batch(batch, start_idx):
        documents, metadatas, ids = [], [], []
        for idx, split in enumerate(batch):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{start_idx + idx}")
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        st.success(f"Processed batch {start_idx // batch_size + 1} of {len(all_splits) // batch_size + 1}")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, all_splits[i:i + batch_size], i) for i in range(0, len(all_splits), batch_size)]
        for future in futures:
            future.result()

    st.success("All data added to the vector store!")

@st.cache_data
def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents."""
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="mistral",  # Use a smaller model
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

@st.cache_data
def re_rank_cross_encoders(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring."""
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

# Initialize session state
if 'process_clicked' not in st.session_state:
    st.session_state.process_clicked = False

def on_process_click():
    """Callback function to update session state when the 'Process' button is clicked."""
    st.session_state.process_clicked = True

if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer")
    st.header("🗣️ RAG Question Answer")

    # Document Upload and Processing
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )
        process = st.button("⚡️ Process", on_click=on_process_click)

        if uploaded_file and st.session_state.process_clicked:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name, batch_size=10)

    # Question and Answer Area
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context, prompt)
        response = call_llm(context=relevant_text, prompt=prompt)
        
        # Stream the response
        response_placeholder = st.empty()
        full_response = ""
        for chunk in response:
            full_response += chunk
            response_placeholder.markdown(full_response)

        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
            #make some changes on sqlite3