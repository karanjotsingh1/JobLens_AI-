# ============================================================
# rag/resume_jd_rag.py
#
# PURPOSE: Build the RAG (Retrieval Augmented Generation) pipeline
#          for parsing resumes and job descriptions.
#
# WHAT THIS FILE DOES:
#   1. Load PDF (resume or JD)
#   2. Split into smart chunks (different sizes for resume vs JD)
#   3. Embed chunks using FREE HuggingFace sentence-transformers
#   4. Store in FAISS vector database
#   5. Calculate JD-Resume match score using cosine similarity
#   6. Extract structured info (skills, projects, experience) via LLM
#
# WHY DIFFERENT CHUNK SIZES FOR RESUME VS JD?
#   Resume → smaller chunks (400 chars) because each bullet point
#             is a separate skill/achievement — we don't want them merged
#   JD     → larger chunks (600 chars) because requirements are often
#             written in multi-sentence paragraphs that need full context
# ============================================================

import os
import sys
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders    import PyPDFLoader
from langchain.text_splitter                  import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores         import FAISS
from langchain.schema                         import Document

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL,
    RESUME_CHUNK_SIZE, RESUME_CHUNK_OVERLAP,
    JD_CHUNK_SIZE,     JD_CHUNK_OVERLAP,
    RAG_TOP_K
)



# ─────────────────────────────────────────────────────────────
# 1. LOAD SENTENCE TRANSFORMER MODEL
# ─────────────────────────────────────────────────────────────
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────
# 2. PDF LOADER — Convert uploaded PDF → list of text chunks
# ─────────────────────────────────────────────────────────────
def load_pdf_to_chunks(pdf_bytes: bytes, doc_type: str = "resume") -> List[Document]:
    """
    Load a PDF from raw bytes (uploaded via Streamlit) and split into chunks.

    Args:
        pdf_bytes: Raw bytes from Streamlit file_uploader
        doc_type:  "resume" or "jd" — controls chunk size

    Returns:
        List of LangChain Document objects (each has .page_content + .metadata)
    """
    # Write bytes to a temporary file so PyPDFLoader can read it
    # (PyPDFLoader needs a real file path, not bytes directly)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Load PDF pages as Documents
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)

    # Choose chunk settings based on document type
    if doc_type == "resume":
        chunk_size    = RESUME_CHUNK_SIZE
        chunk_overlap = RESUME_CHUNK_OVERLAP
    else:  # "jd"
        chunk_size    = JD_CHUNK_SIZE
        chunk_overlap = JD_CHUNK_OVERLAP

    # Split pages into smaller chunks using RecursiveCharacterTextSplitter
    # It tries to split on \n\n first, then \n, then spaces — keeps semantic units intact
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "•", "–", "-", " ", ""],  # bullet-aware splitting
    )

    chunks = splitter.split_documents(pages)

    # Tag each chunk with its document type in metadata
    for chunk in chunks:
        chunk.metadata["doc_type"] = doc_type

    return chunks



# ─────────────────────────────────────────────────────────────
# 3. BUILD FAISS VECTOR STORE
# ─────────────────────────────────────────────────────────────
def build_vectorstore(chunks, embedding_model):
    """
    Build FAISS vector store from document chunks.
    """

    print("Starting FAISS.from_documents()...")

    return FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

# ─────────────────────────────────────────────────────────────
# 4. CALCULATE MATCH SCORE — Resume vs JD cosine similarity
# ─────────────────────────────────────────────────────────────
def calculate_match_score(
    resume_chunks : List[Document],
    jd_chunks     : List[Document],
    embeddings    : HuggingFaceEmbeddings
) -> Tuple[float, List[str]]:
    """
    Calculate how well a resume matches a job description.

    METHOD:
    1. Embed the full JD text as one query vector
    2. Search against resume chunks using cosine similarity
    3. Average similarity of top-K matches → match score (0-100%)

    Returns:
        match_score:     float 0-100 (percentage match)
        matching_chunks: List of resume snippets that matched best
    """
    # Combine all JD chunks into one query string
    jd_full_text = "\n".join([c.page_content for c in jd_chunks])

    # Build FAISS from resume chunks
    resume_store = build_vectorstore(resume_chunks, embeddings)
    print("✅ Vector store built")

    # Search resume for sections most similar to the JD
    retriever = resume_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RAG_TOP_K}
    )
    print("✅ Retriever created")

    # Retrieve top matching resume chunks
    matched_docs = retriever.invoke(jd_full_text)
    print("✅ Retrieved documents:", len(matched_docs))

    # Calculate similarity scores manually using dot product
    jd_vector = embeddings.embed_query(jd_full_text[:2000])
    print("✅ JD embedded")

    matched_scores = []

    for i, doc in enumerate(matched_docs):
        print(f"➡️ Embedding resume chunk {i}")

        chunk_vector = embeddings.embed_query(doc.page_content)

        print(f"✅ Resume chunk {i} embedded")

        similarity = sum(a * b for a, b in zip(jd_vector, chunk_vector))
        matched_scores.append(max(0.0, similarity))

    # Average similarity across top matches, scale to 0-100
    avg_similarity = sum(matched_scores) / len(matched_scores) if matched_scores else 0
    match_score    = min(round(avg_similarity * 100, 1), 100.0)

    # Return matching snippets for display in UI
    matching_snippets = [doc.page_content for doc in matched_docs]

    return match_score, matching_snippets


# ─────────────────────────────────────────────────────────────
# 5. RAG RETRIEVER — For chatbot & gap analysis queries
# ─────────────────────────────────────────────────────────────
def get_retriever(vectorstore: FAISS, search_type: str = "mmr"):
    """
    Return a retriever from the vectorstore.

    We use MMR (Maximal Marginal Relevance) search type instead of
    plain similarity. MMR ensures retrieved chunks are:
    - Relevant to the query
    - Diverse from each other (avoids returning same info twice)

    This is crucial for resumes where similar bullet points exist.
    """
    return vectorstore.as_retriever(
        search_type="mmr",               # MMR for diverse, relevant results
        search_kwargs={
            "k": RAG_TOP_K,
            "fetch_k": RAG_TOP_K * 3,   # Fetch more, then re-rank for diversity
            "lambda_mult": 0.7,          # 0=max diversity, 1=max relevance
        }
    )


# ─────────────────────────────────────────────────────────────
# 6. EXTRACT STRUCTURED INFO from resume using LLM + RAG
# ─────────────────────────────────────────────────────────────
def extract_resume_info(resume_text: str, llm) -> dict:
    """
    Use LLM to extract structured information from raw resume text.

    Returns a dict with:
        skills:     list of technical skills mentioned
        experience: years of experience + roles held
        projects:   list of projects with descriptions
        education:  education details
    """
    # This prompt tells the LLM exactly what to extract and how to format it
    extraction_prompt = f"""
You are an expert resume parser. Extract the following information from this resume.
Be thorough and precise. Do not make up information — only extract what is written.

RESUME TEXT:
{resume_text[:4000]}

Extract and return as a structured summary with these EXACT sections:

SKILLS: (list all technical skills, tools, languages, frameworks)
EXPERIENCE: (job titles, companies, durations)
PROJECTS: (project names with 1-line descriptions of what they did + tech used)
EDUCATION: (degrees, institutions, years)
CERTIFICATIONS: (any certifications or online courses mentioned)

Be specific and complete. Do not skip any technical skill or project.
"""

    # Call the LLM to extract structured info
    response = llm.invoke(extraction_prompt)
    return {"raw_extraction": response.content}
