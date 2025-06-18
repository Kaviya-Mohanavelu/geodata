import os
import sys
import streamlit as st
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# -----------------------------
# 🔐 API Configuration
# -----------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyALWwif_Sw8e6DX4tgOFrBzHBciYo9LQ7g"
Entrez.api_key = "2546a5ffeb7525d1d3edb654c2f618dd0709"
Entrez.email = "kaviyamohanavelu@gmail.com"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# -----------------------------
# 🤖 Gemini Model Selection
# -----------------------------
def get_best_supported_model():
    preferred_models = [
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.0-pro"
    ]
    available_models = genai.list_models()
    supported = [
        model.name for model in available_models
        if "generateContent" in getattr(model, "supported_generation_methods", [])
    ]
    for preferred in preferred_models:
        if preferred in supported:
            return preferred
    raise Exception("❌ No supported Gemini models found.")

# -----------------------------
# 📡 GEO Data Functions
# -----------------------------
def fetch_geo_data(term, organism=None, platform=None):
    query = term
    if organism:
        query += f" AND {organism}[Organism]"
    if platform:
        query += f" AND {platform}[Platform]"

    handle = Entrez.esearch(db="gds", term=query, usehistory="y", retmax=0)
    record = Entrez.read(handle)
    handle.close()

    total = int(record["Count"])
    if total == 0:
        return [], 0

    ids = []
    webenv = record["WebEnv"]
    query_key = record["QueryKey"]
    for start in range(0, total, 500):
        handle = Entrez.esearch(
            db="gds", term=query, retstart=start, retmax=500,
            usehistory="y", webenv=webenv, query_key=query_key
        )
        chunk = Entrez.read(handle)
        ids.extend(chunk["IdList"])
        handle.close()
    return ids, total

def fetch_geo_metadata(id_list):
    results = []
    for gid in tqdm(id_list, desc="📦 Fetching metadata"):
        try:
            summary = Entrez.esummary(db="gds", id=gid, retmode="xml")
            record = Entrez.read(summary)[0]
            results.append({
                "Title": record.get("title", "N/A"),
                "Summary": record.get("summary", "N/A"),
                "Organism": record.get("taxon", "N/A"),
                "Experiment Type": record.get("gdsType", "N/A"),
                "Platform": record.get("GPL", "N/A"),
                "Sample Size": record.get("n_samples", "N/A"),
            })
        except Exception as e:
            print(f"❌ Error for {gid}: {e}")
    return results

def create_docs_from_metadata(metadata):
    docs = []
    for entry in metadata:
        content = "\n".join(f"{k}: {v}" for k, v in entry.items())
        docs.append(Document(page_content=content))
    return docs

# -----------------------------
# 🚀 Streamlit App
# -----------------------------
st.set_page_config(page_title="🧬 GEO Explorer", layout="centered")
st.title("🧬 AI-Powered GEO Explorer")

query = st.text_input("🔍 Enter keyword, accession number, or author/contributor name")
organism = st.text_input("🧬 Organism (optional)")
platform = st.text_input("📟 Platform (optional)")

if st.button("Search GEO"):
    if not query:
        st.warning("Please enter a search query.")
        st.stop()

    with st.spinner("🔎 Searching GEO database..."):
        ids, total = fetch_geo_data(query, organism, platform)

    if not ids:
        st.error("❌ No datasets found. Try a different query.")
        st.stop()

    metadata = fetch_geo_metadata(ids)
    if not metadata:
        st.error("❌ No metadata retrieved.")
        st.stop()

    st.success(f"✅ Fetched {len(metadata)} datasets.")
    st.dataframe(pd.DataFrame(metadata))

    docs = create_docs_from_metadata(metadata)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory="./chromadb"
    )

    try:
        model_name = get_best_supported_model()
        llm = ChatGoogleGenerativeAI(model=model_name)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    except Exception as e:
        st.error(f"❌ QA failed: {e}")
        st.stop()

    st.subheader("❓ Ask a question about the datasets")
    question = st.text_input("Enter your question here")

    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
            st.stop()
        try:
            answer = qa_chain.run(question)
            st.markdown("### 🧠 Answer")
            st.info(answer)

            retrieved_docs = vectordb.similarity_search(question, k=3)
            st.markdown("### 📚 Top 3 Relevant Dataset Descriptions")
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**Doc {i+1}:**\n{doc.page_content}")
        except Exception as e:
            st.error(f"❌ Error during QA: {e}")
