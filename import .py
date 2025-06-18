
import os
import time
import streamlit as st
from Bio import Entrez
from tqdm import tqdm
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# -----------------------------
# 🔐 Configuration
# -----------------------------
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
Entrez.api_key = st.secrets["NCBI_API_KEY"]
Entrez.email = "kaviyamohanavelu@gmail.com"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# -----------------------------
# 🤖 Select Gemini Model
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
# 🔍 GEO Functions
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

    if int(record["Count"]) == 0:
        return [], 0

    webenv = record["WebEnv"]
    query_key = record["QueryKey"]
    total = int(record["Count"])

    ids = []
    for start in range(0, total, 500):
        time.sleep(0.4)
        handle = Entrez.esearch(
            db="gds", term=query, retstart=start, retmax=500, usehistory="y",
            webenv=webenv, query_key=query_key
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
            st.warning(f"❌ Error for {gid}: {e}")
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
st.set_page_config(page_title="🧬 GEO + Gemini Explorer")
st.title("🧬 AI-Powered GEO Explorer: Biomedical Insights with RAG")

query = st.text_input("🔎 Enter keyword, accession number, or author/contributor name")
organism = st.text_input("🧬 Filter by organism (optional)")
platform = st.text_input("📟 Filter by platform (optional)")

if st.button("Search GEO Datasets"):
    with st.spinner("🔍 Searching GEO database..."):
        ids, total = fetch_geo_data(query, organism, platform)

    if total == 0:
        st.error("❌ No datasets found. Try a different query.")
    else:
        st.success(f"✅ Found {total} matching GEO datasets.")
        metadata = fetch_geo_metadata(ids)
        st.dataframe(metadata)

        docs = create_docs_from_metadata(metadata)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(
            docs, embedding=embeddings, persist_directory="./chromadb"
        )

        model_name = get_best_supported_model()
        llm = ChatGoogleGenerativeAI(model=model_name)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=vectordb.as_retriever()
        )

        st.session_state.qa_chain = qa_chain
        st.session_state.vectordb = vectordb
        st.session_state.active = True

if st.session_state.get("active"):
    st.subheader("❓ Ask a question about the datasets")
    question = st.text_input("Enter your biomedical question")
    if st.button("Get Answer"):
        with st.spinner("💬 Generating answer using Gemini..."):
            try:
                answer = st.session_state.qa_chain.run(question)
                st.success("🧠 Answer:")
                st.write(answer)

                st.markdown("### 📚 Top Relevant Datasets:")
                retrieved_docs = st.session_state.vectordb.similarity_search(question, k=3)
                for doc in retrieved_docs:
                    st.markdown("---")
                    st.markdown(doc.page_content)
            except Exception as e:
                st.error(f"❌ QA failed: {e}")
