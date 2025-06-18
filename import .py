import os
import sys
import streamlit as st
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# -----------------------------
# 🔐 Configuration
# -----------------------------
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
Entrez.api_key = st.secrets["NCBI_API_KEY"]
Entrez.email = st.secrets["EMAIL"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# -----------------------------
# 🤖 Gemini Model Fallback
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
# GEO Query Functions
# -----------------------------
def fetch_geo_data(term, organism=None, platform=None):
    query = term
    if organism and organism.lower() != "no":
        query += f" AND {organism}[Organism]"
    if platform and platform.lower() != "no":
        query += f" AND {platform}[Platform]"

    handle = Entrez.esearch(db="gds", term=query, usehistory="y", retmax=0)
    record = Entrez.read(handle)
    handle.close()

    webenv = record.get("WebEnv")
    query_key = record.get("QueryKey")
    total = int(record.get("Count", 0))

    ids = []
    for start in range(0, total, 500):
        handle = Entrez.esearch(
            db="gds", term=query, retstart=start, retmax=500, usehistory="y",
            webenv=webenv, query_key=query_key
        )
        chunk = Entrez.read(handle)
        ids.extend(chunk.get("IdList", []))
        handle.close()
    return ids, total

def fetch_geo_metadata(id_list):
    results = []
    for gid in tqdm(id_list, desc="📦 Fetching metadata"):
        try:
            summary = Entrez.esummary(db="gds", id=gid, retmode="xml")
            record = Entrez.read(summary)[0]
            sample_size = str(record.get("n_samples", "N/A"))
            results.append({
                "Title": record.get("title", "N/A"),
                "Summary": record.get("summary", "N/A"),
                "Organism": record.get("taxon", "N/A"),
                "Experiment Type": record.get("gdsType", "N/A"),
                "Platform": record.get("GPL", "N/A"),
                "Sample Size": sample_size,
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
# Streamlit UI
# -----------------------------
st.title("🧬 AI-Powered GEO Explorer: Biomedical Insights with RAG")
query = st.text_input("🔎 Enter keyword, accession number, or author/contributor name")
organism = st.text_input("🧬 Filter by organism (optional)")
platform = st.text_input("📟 Filter by platform (optional)")

if st.button("🔍 Search GEO Datasets"):
    if not query:
        st.warning("Please enter a query term.")
    else:
        with st.spinner("🔎 Searching GEO..."):
            ids, total = fetch_geo_data(query, organism, platform)
            st.write(f"📊 Matching GEO Datasets: {total}")

            if not ids:
                st.error("❌ No datasets found. Try a different query.")
            else:
                metadata = fetch_geo_metadata(ids)
                st.success(f"✅ Datasets fetched: {len(metadata)}")
                df = pd.DataFrame(metadata)
                st.dataframe(df)

                # LangChain Docs
                docs = create_docs_from_metadata(metadata)
                if not docs:
                    st.error("❌ No valid documents for vector store.")
                else:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectordb = FAISS.from_documents(
                        docs,
                        embedding=embeddings
                    )
                    model_name = get_best_supported_model()
                    llm = ChatGoogleGenerativeAI(model=model_name)
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

                    st.session_state["qa_chain"] = qa_chain
                    st.session_state["vectordb"] = vectordb

# -----------------------------
# QA Interface
# -----------------------------
if "qa_chain" in st.session_state:
    st.subheader("💬 Ask a question about the datasets")
    user_question = st.text_input("❓ Your Question")
    if st.button("💡 Get Answer"):
        with st.spinner("🤖 Generating answer..."):
            try:
                answer = st.session_state["qa_chain"].run(user_question)
                st.write("🧠 *Answer:*", answer)

                retrieved_docs = st.session_state["vectordb"].similarity_search(user_question, k=3)
                st.markdown("### 📚 Top Relevant Documents")
                for doc in retrieved_docs:
                    st.code(doc.page_content)
            except Exception as e:
                st.error(f"❌ QA failed: {e}")
