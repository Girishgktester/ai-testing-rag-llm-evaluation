"""Small Streamlit UI for webpage RAG using OpenAI (max 1000 completion tokens)."""

import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("USER_AGENT", "webpage-rag-ui")

OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
MAX_OUTPUT_TOKENS = 1000


def load_openai_env() -> Path:
    cwd = Path.cwd()
    env_candidates = [
        cwd / ".env",
        cwd / "notebooks" / ".env",
        cwd.parent / "notebooks" / ".env",
        cwd / "RAG" / ".env",
        Path(__file__).resolve().parent.parent / "notebooks" / ".env",
    ]
    env_file = next((p for p in env_candidates if p.exists()), None)
    if env_file is None:
        raise FileNotFoundError("No .env found. Expected notebooks/.env with OPENAI_API_KEY.")
    load_dotenv(env_file, override=True)
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or "paste_your_key" in key:
        raise ValueError(f"Set OPENAI_API_KEY in {env_file}.")
    return env_file


@st.cache_resource
def get_embeddings():
    load_openai_env()
    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)


@st.cache_resource
def get_llm():
    load_openai_env()
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0.5,
        max_tokens=MAX_OUTPUT_TOKENS,
    )


def load_and_index(url: str, embeddings) -> tuple[Chroma, int]:
    documents = WebBaseLoader([url]).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    # New collection each load so we never mix Ollama (768) with OpenAI (1536) vectors.
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chromadb.EphemeralClient(),
        collection_name="webpage_rag_openai",
    )
    return store, len(chunks)


st.set_page_config(page_title="Webpage RAG", layout="centered")
st.title("Webpage RAG")
st.caption(
    f"OpenAI {OPENAI_CHAT_MODEL}. Answers use retrieved page text only. "
    f"Completion is capped at {MAX_OUTPUT_TOKENS} tokens."
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.loaded_url = ""
    st.session_state.chunk_count = 0

try:
    env_file = load_openai_env()
    st.sidebar.success(f"OPENAI_API_KEY loaded from {env_file.name}")
    st.sidebar.write(f"Chat model: `{OPENAI_CHAT_MODEL}`")
    st.sidebar.write(f"Max output tokens: `{MAX_OUTPUT_TOKENS}`")
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

with st.form("load_form"):
    url = st.text_input(
        "Webpage URL",
        value="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    )
    load_clicked = st.form_submit_button("Load page", type="primary")

if load_clicked:
    if not url.strip():
        st.error("Enter a URL first.")
    else:
        with st.spinner("Fetching page, splitting, and embedding..."):
            store, n_chunks = load_and_index(url.strip(), get_embeddings())
        st.session_state.vector_store = store
        st.session_state.loaded_url = url.strip()
        st.session_state.chunk_count = n_chunks
        st.success(f"Loaded {n_chunks} chunks from {url.strip()}")

if st.session_state.vector_store is not None:
    st.info(f"Ready: {st.session_state.chunk_count} chunks from {st.session_state.loaded_url}")

with st.form("ask_form"):
    question = st.text_input(
        "Question",
        value="What is retrieval-augmented generation?",
    )
    ask_clicked = st.form_submit_button("Ask")

if ask_clicked:
    if st.session_state.vector_store is None:
        st.error("Load a page first.")
    elif not question.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Searching and generating an answer..."):
            docs = st.session_state.vector_store.similarity_search(question.strip(), k=3)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = f"""Answer the question using only the context below.
If the answer is not present in the context, say: I don't know based on the webpage.

Context:
{context}

Question: {question.strip()}
Answer:"""
            response = get_llm().invoke(prompt)
        st.subheader("Answer")
        st.write(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            st.caption(
                f"Token usage — input: {usage.get('input_tokens')}, "
                f"output: {usage.get('output_tokens')} (cap {MAX_OUTPUT_TOKENS}), "
                f"total: {usage.get('total_tokens')}"
            )
        with st.expander("Retrieved chunks"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)
