# Retrieval-Augmented Generation (RAG) Question Answering System for Vietnamese Documents

<p align="right"> Lang: <b>EN</b> | <a href="docs/locales/README.vi.md"><b>VI</b></a> </p>

## 1. Overview & Architecture

- **RAG (Retrieval-Augmented Generation):** A technique that combines information retrieval from an internal document repository with a Large Language Model (LLM) to generate accurate, evidence-based answers and reduce hallucinations.
- **Domain-Specific Corpus:** A collection of specialized documents, such as legal texts, financial records, or corporate documents, used as the knowledge base for a RAG system.
- **Agentic RAG:** An advanced RAG architecture in which an agent can plan, perform multi-step retrieval, invoke tools, and verify context completeness before answering.

## 2. Data Preprocessing & Vietnamese Language Specifics

- **Vietnamese Word Segmentation:** Vietnamese compound words consist of multiple syllables separated by spaces. Word segmentation combines syllables into lexical units, such as converting "học sinh" into "học_sinh", helping embedding models capture semantic relationships.
  - **PyVi:** Fast, with moderate accuracy.
  - **Underthesea:** Provides a balance between accuracy and performance.
  - **VnCoreNLP:** Uses linguistic processing techniques, including CRF-based models, for word segmentation.
  - **coccoc-tokenizer:** A high-performance C++ tokenizer.
- **Tokenizer Overhead (Token Count Inflation):** Vietnamese text may produce more tokens than English text, depending on the tokenizer and model. This can increase computational costs and latency.
- **Text Chunking:** Splitting long documents into smaller segments, typically 256–512 tokens, with approximately 10–20% overlap to preserve contextual continuity.
  - **Word-Aware Chunking:** Preserves Vietnamese word boundaries and avoids splitting multi-syllabic lexical units across chunks.

## 3. Retrieval Layer & Vector Space

- **Bi-Encoder:** An architecture that independently encodes queries and document passages into fixed-dimensional vectors in a shared embedding space.
- **Dense Retrieval:** Retrieves semantically similar passages using vector similarity measures such as cosine similarity. Example models include `Vietnamese_Embedding`, `bkai-foundation-models/vietnamese-bi-encoder`, and `bge-m3`.
- **Sparse Retrieval:** Uses keyword matching and term-weighting algorithms such as BM25 to retrieve passages containing relevant terms.
- **Hybrid Search:** Combines dense semantic retrieval and sparse keyword retrieval using weighted score fusion or Reciprocal Rank Fusion (RRF).
- **Multi-Vector / Late Interaction (ColBERT):** Retains token-level vectors to enable fine-grained query-document matching at the cost of additional computation and storage.
- **Vector Database:** Systems for storing embeddings and performing similarity searches, including FAISS, Chroma, Weaviate, and Qdrant.

## 4. Reranking Layer

- **Cross-Encoder / Reranker:** Jointly processes a query and a document passage to estimate their relevance through detailed token-level interactions.
- **Vietnamese Reranker:** A reranking model adapted for Vietnamese, potentially based on `bge-reranker-v2-m3`, used to reorder retrieved candidates and select the most relevant passages.
- **Negative Mining:** Selects incorrect examples during training to improve retrieval and ranking quality.
  - **Hard Negatives:** Incorrect passages that closely resemble relevant passages.
  - **Semi-Hard Negatives:** Incorrect passages with moderate similarity, helping balance training difficulty.

## 5. Text Generation & Large Language Models (LLMs)

- **Vietnamese LLMs:** Models that synthesize retrieved context into natural-language answers. Examples include `SeaLLM-7B-v2`, `Vistral-7B`, `VinaLLaMA`, `Qwen2.5`, and `GPT-4o`.
- **System Prompts & Guardrails:** Instructions that define the model's role, require evidence-based answers and source citations, and direct it to abstain when sufficient evidence is unavailable.
- **Hallucination:** The generation of information that is unsupported by or inconsistent with source documents.
- **Dialect Robustness:** The ability to handle vocabulary, pronoun, and expression differences across Northern, Central, and Southern Vietnamese dialects.

## 6. System Evaluation Framework & Metrics

- **Retrieval Evaluation Metrics:**
  - **Recall@K:** The proportion of relevant documents retrieved within the top K results.
  - **MRR (Mean Reciprocal Rank):** Measures the average reciprocal rank of the first relevant result across queries.
  - **nDCG@K (Normalized Discounted Cumulative Gain):** Evaluates ranking quality using graded relevance and the positions of relevant results.
  - **Exist@m:** The proportion of queries for which at least one passage containing the correct answer appears among the top m candidates.
- **RAGAS / RAG Triad Evaluation Metrics:**
  - **Context Relevance:** Measures how useful the retrieved context is for answering the query.
  - **Answer Faithfulness / Groundedness:** Measures whether claims in the generated answer are supported by the retrieved context.
  - **Answer Relevance:** Measures how well the generated answer addresses the user's original question and intent.
