# RAG webpage notebook: what each step does

This notebook (`RagForPDF.ipynb`) is a **Retrieval-Augmented Generation** pipeline. The goal is:

> Read a webpage you give it, store that text as searchable chunks, then answer questions using only those chunks — not from the LLM’s general memory.

Without RAG, the local model (`qwen3:8b`) might guess or hallucinate. With RAG, it must look at retrieved webpage text first.

```
 INDEX PATH (run once per URL)              QUERY PATH (every question)

 +------------------+                       +------------------+
 | 1. URL you set   |                       | 7. Your question |
 +--------+---------+                       +--------+---------+
          |                                          |
          v                                          v
 +------------------+                       +------------------+
 | 2. Load webpage  |                       | 8. Embed question|
 |    WebBaseLoader |                       |    same model    |
 +--------+---------+                       +--------+---------+
          |                                          |
          v                                          v
 +------------------+                       +------------------+
 | 3. Split text    |                       | 9. Similarity    |
 |    into chunks   |                       |    search k=3    |
 +--------+---------+                       +--------+---------+
          |                                          |
          v                                          |
 +------------------+                                |
 | 4. Embed chunks  |                                |
 |    nomic-embed   |                                |
 +--------+---------+                                |
          |                                          |
          v                                          |
 +------------------+                                |
 | 5. Save vectors  |<-------------------------------+
 |    Chroma disk   |
 +--------+---------+
          |
          v
 +------------------+
 | 6. Ready to ask  |
 +------------------+
          |
          v
 +------------------+
 | 10. Build prompt |
 |     context+Q    |
 +--------+---------+
          |
          v
 +------------------+
 | 11. LLM answers  |
 |     qwen3:8b     |
 +--------+---------+
          |
          v
 +------------------+
 | 12. Printed text |
 +------------------+
```

---

## 1. Setup: env, packages, LLM

**What:** Load `.env`, create `ChatOllama` (`qwen3:8b` at `http://localhost:11434`).

**Why:** The answer model must already exist before you ask a question. Ollama is the local LLM. `.env` is for any keys you might add later.

**Trying to do:** Connect to a local chat model. This step does **not** read the webpage.

---

## 2. Load the webpage

**What:** `urls = [...]` plus `WebBaseLoader`. It fetches the HTML and turns it into LangChain `Document` objects (plain text + metadata like `source`).

**Why:** The rest of the pipeline cannot search a live URL. It needs the page content in memory as text.

**Trying to do:** “Whatever URL I put here, download it and extract readable text.” Add more URLs in the list to load several pages.

---

## 3. Split into chunks

**What:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.

**Why:** A full Wikipedia page is too long to embed or stuff into one prompt well. Small overlapping pieces let search find the *relevant paragraph*, not the whole article.

**Trying to do:** Cut the page into ~1000-character windows that overlap by 200 so a sentence is not split awkwardly across two chunks.

---

## 4. Embed the chunks

**What:** `OllamaEmbeddings(model="nomic-embed-text")`, sent in batches of 8.

**Why:** Search is not keyword-only. An embedding is a list of numbers that represent *meaning*. Similar text → similar numbers. Batching avoids Ollama failing on a huge list in one request.

**Trying to do:** Convert each chunk into a vector so “retrieval-augmented generation” can match a chunk even if the wording is different.

The two `embed_query` prints are a smoke test: they show vector length so you know embeddings work.

---

## 5. Vector store (Chroma + `persist_directory`)

**What:** `Chroma.from_documents(..., persist_directory="./chroma_langchain_db_web")`.

**Why:** Chroma stores (chunk text + vector) so you can search later. `persist_directory` writes that index to disk. Without it, the index dies when the kernel restarts.

**Trying to do:** Save the knowledge base next to the notebook so the next cell can reload it instead of re-embedding the whole page.

The following cell opens the **same folder** again:

```python
Chroma(persist_directory="./chroma_langchain_db_web", embedding_function=embeddings)
```

That is reload, not a second copy of the webpage.

---

## 6. Retrieve relevant chunks

**What:** `similarity_search(question, k=3)`.

**Why:** The LLM should only see the few chunks that match the question. Sending the entire page is slow and noisy.

**Trying to do:** Embed the question with the **same** embedding model, compare it to stored chunk vectors, return the 3 nearest chunks.

---

## 7. Generate an answer from context

**What:** Join those 3 chunks into `context`, put them in a prompt: “Answer using only the context. If it is not there, say you don’t know.” Then `llm.invoke(prompt)`.

**Why:** This is the “augmented generation” part. The model is constrained to the webpage snippets, which reduces hallucination.

**Trying to do:** Produce an answer grounded in the retrieved text, not from the model’s training data.

---

## 8. Retriever + RetrievalQA chain

**What:** `vector_store.as_retriever(k=3)` and a LangChain chain:

```
question -> retriever -> format chunks -> prompt -> llm -> string
```

**Why:** Same as steps 6–7, but as one reusable pipeline. You call `retrieval_qa_chain.invoke(question)` instead of wiring search and prompt by hand.

**Trying to do:** A compact “ask anything about the loaded page” API for later cells.

---

## What you change vs what you leave

| You change | You usually leave |
|---|---|
| `urls` (the webpage) | splitter sizes unless chunks are too big/small |
| `question` (what you ask) | embedding model (`nomic-embed-text`) |
| `k` (how many chunks) | Chroma folder name, unless you want a fresh index |

If you change the URL, run **load → split → embed → `from_documents` again**. Old Chroma files still hold the previous page until you rebuild.

---

## End-to-end in one sentence

**Fetch the page → chop it → turn chunks into vectors → store them → for each question, find the closest chunks → ask the local LLM to answer only from those chunks.**
