import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Title
st.set_page_config(page_title="RAG Demo - No API Key", layout="wide")
st.title(" Local RAG Demo (No API Key Needed)")

# Load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    text_column = st.selectbox("Select text column for context:", df.columns)
    docs = df[text_column].astype(str).tolist()

    # Embed model (lightweight + no API key)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Precompute embeddings
    st.info("Encoding text data... (only runs once)")
    doc_embeddings = model.encode(docs, show_progress_bar=True)

    # User query
    query = st.text_input("Enter your question:")
    if query:
        query_emb = model.encode([query])
        sims = cosine_similarity(query_emb, doc_embeddings)[0]
        top_idx = sims.argsort()[-3:][::-1]

        st.subheader(" Top Relevant Results")
        for i in top_idx:
            st.markdown(f"**Context:** {docs[i]}")
            st.markdown(f"**Similarity:** {sims[i]:.3f}")
            st.markdown("---")
else:
    st.info("Please upload a CSV file to get started.")
