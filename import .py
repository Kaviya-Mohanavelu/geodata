import os
import streamlit as st
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
st.set_page_config(page_title="🧬 GEO Explorer", layout="wide")
st.title("🧬 AI-Powered GEO Explorer: Biomedical Insights with RAG")

# Load Gemini API key
api_key = st.secrets["AIzaSyALWwif_Sw8e6DX4tgOFrBzHBciYo9LQ7g"]
genai.configure(api_key=api_key)

# -----------------------------
# 🧠 LLM + Embedding Setup
# -----------------------------
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 🔍 GEO Dataset Fetch Function
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_geo_data(query, organism=None, platform=None, max_results=100):
    Entrez.email = "kaviyamohanavelu@gmail.com"
    search_term = query
    if organism:
        search_term += f" AND {organism}[Organism]"
    if platform:
        search_term += f" AND {platform}[Platform]"

    handle = Entrez.esearch(db="gds", term=search_term, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    datasets = []

    if not ids:
        return datasets

    for i in tqdm(range(0, len(ids), 20)):
        chunk = ids[i:i+20]
        handle = Entrez.esummary(db="gds", id=",".join(chunk))
        summaries = Entrez.read(handle)
        handle.close()

        for summary in summaries["DocumentSummarySet"]["DocumentSummary"]:
            meta = {
                "title": summary["title"],
                "summary": summary["summary"],
                "platform": summary["GPL"],
                "sample_size": summary["n_samples"],
                "organism": summary["taxon"],
                "type": summary["gdsType"]
            }
            datasets.append(meta)

    return datasets

# -----------------------------
# 🧠 Build FAISS Vector Store
# -----------------------------
def build_vector_store(datasets):
    docs = []
    for entry in datasets:
        content = f"{entry['title']}\n{entry['summary']}\n{entry['organism']}\n{entry['platform']}\n{entry['type']}\nSamples: {entry['sample_size']}"
        doc = Document(page_content=content, metadata=entry)
        docs.append(doc)

    if not docs:
        return None, None

    vectordb = FAISS.from_documents(docs, embedding=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
    return qa, docs

# -----------------------------
# 🧪 Streamlit UI
# -----------------------------
with st.form("geo_query_form"):
    query = st.text_input("🔎 Enter keyword, accession number, or author/contributor name", "osteoclast AND mirna")
    col1, col2 = st.columns(2)
    with col1:
        organism = st.text_input("🧬 Organism (optional)")
    with col2:
        platform = st.text_input("🔬 Platform (optional)")
    submitted = st.form_submit_button("🚀 Search GEO")

if submitted:
    with st.spinner("🔍 Fetching GEO datasets..."):
        datasets = fetch_geo_data(query, organism, platform)

    if datasets:
        st.success(f"✅ {len(datasets)} datasets found.")
        st.dataframe(datasets)

        qa_chain, docs = build_vector_store(datasets)

        if not qa_chain:
            st.error("❌ No valid documents found to build the vector store.")
        else:
            st.markdown("### 💬 Ask a Question about the Datasets")
            user_q = st.text_input("Enter your question")

            if user_q:
                with st.spinner("🤖 Generating answer..."):
                    try:
                        result = qa_chain.run(user_q)
                        st.markdown("### 🧠 Answer")
                        st.success(result)
                    except Exception as e:
                        st.error(f"❌ QA failed: {e}")
    else:
        st.error("❌ No datasets found. Try a different query.")
