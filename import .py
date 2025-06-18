import os
from Bio import Entrez
from tqdm import tqdm
import streamlit as st
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# --------------------------
# 🔐 Configuration
# --------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyALWwif_Sw8e6DX4tgOFrBzHBciYo9LQ7g"
Entrez.api_key = "2546a5ffeb7525d1d3edb654c2f618dd0709"
Entrez.email = "kaviyamohanavelu@gmail.com"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --------------------------
# 🎨 Streamlit UI Setup
# --------------------------
st.set_page_config(page_title="AI-Powered GEO Explorer", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #4A90E2;
            margin-top: -30px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #7B8D93;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 20px;
        }
        .stTextInput>div>div>input {
            background-color: #f2faff;
            border: 1px solid #d6e9f8;
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    <div class="main-title"> cd cd "C:\Users\udhayan\Documents\java\K"AI-Powered GEO Explorer</div>
    <div class="subtitle">🔍 Biomedical Insights with Retrieval-Augmented Generation</div>
""", unsafe_allow_html=True)

# --------------------------
# 🔍 Model Selection
# --------------------------
@st.cache_data(show_spinner=False)
def get_best_supported_model():
    preferred_models = [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
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

# --------------------------
# 🧬 GEO Data Functions
# --------------------------
@st.cache_data(show_spinner=False)
def fetch_geo_data(term, organism=None, platform=None):
    if not term or not term.strip():
        raise ValueError("Search term cannot be empty.")
    
    query = term.strip()
    if organism and organism.lower() != "no":
        query += f" AND {organism.strip()}[Organism]"
    if platform and platform.lower() != "no":
        query += f" AND {platform.strip()}[Platform]"

    handle = Entrez.esearch(db="gds", term=query, usehistory="y", retmax=0)
    record = Entrez.read(handle)
    handle.close()

    webenv = record["WebEnv"]
    query_key = record["QueryKey"]
    total = int(record["Count"])

    ids = []
    for start in range(0, total, 500):
        handle = Entrez.esearch(
            db="gds", term=query, retstart=start, retmax=500, usehistory="y",
            webenv=webenv, query_key=query_key
        )
        chunk = Entrez.read(handle)
        ids.extend(chunk["IdList"])
        handle.close()
    return ids, total

@st.cache_data(show_spinner=False)
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
                "Sample Size": str(record.get("n_samples", "N/A")),
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

# --------------------------
# 🔍 Input Section
# --------------------------
st.markdown("---")
st.subheader("📥 Input GEO Search Query")
query = st.text_input("🔍 Enter keyword, accession number, or author/contributor name")
organism = st.text_input("🧬 Organism (optional)")
platform = st.text_input("📟 Platform (optional)")
is_name_search = st.checkbox("🔍 Is the query an author/contributor name?")

# --------------------------
# 🔍 Search Execution
# --------------------------
if st.button("Search GEO"):
    if not query.strip():
        st.warning("⚠️ Please enter a search query.")
        st.stop()

    with st.spinner("🔍 Searching GEO..."):
        try:
            if query.upper().startswith("GSE") or query.upper().startswith("GDS"):
                ids = [query.upper()]
                total = 1
            else:
                qterm = f"{query}[Submitter] OR {query}[Contributor]" if is_name_search else query
                ids, total = fetch_geo_data(qterm, organism, platform)

            if total == 0:
                st.warning("⚠️ No datasets found.")
                st.stop()

            metadata = fetch_geo_metadata(ids)
            st.session_state["metadata"] = metadata

            st.success(f"📊 Found {total} dataset(s).")
            st.dataframe(metadata)

            docs = create_docs_from_metadata(metadata)
            vectordb = Chroma.from_documents(
                docs,
                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                persist_directory="./chromadb"
            )
            st.session_state["vectordb"] = vectordb
            st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(model=get_best_supported_model()),
                retriever=vectordb.as_retriever()
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# --------------------------
# 🤖 QA Section
# --------------------------
if "qa_chain" in st.session_state and "metadata" in st.session_state:
    st.markdown("---")
    st.subheader("🤖 Ask Questions About the Fetched GEO Datasets")
    question = st.text_input("❓ Type your question below")

    if st.button("Ask Gemini"):
        if not question.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("🤖 Generating answer..."):
                try:
                    answer = st.session_state["qa_chain"].run(question)
                    st.success("✅ Gemini Answer")
                    st.write(answer)

                    docs = st.session_state["vectordb"].similarity_search(question, k=3)
                    with st.expander("📚 Top 3 Source Documents"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Doc {i+1}:**")
                            st.code(doc.page_content)

                except Exception as e:
                    st.error(f"⚠️ Error during question answering: {e}")
