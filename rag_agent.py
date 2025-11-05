import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the small embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build RAG-like agent
def build_agent(data_path="business_data.csv"):
    # Load CSV data
    df = pd.read_csv(data_path)

    # Convert rows to descriptive text
    docs = [
        f"Month: {row['Month']}, Sales: {row['Sales (INR)']}, Expenses: {row['Expenses (INR)']}, "
        f"Customers: {row['Customers']}, Inventory Cost: {row['Inventory Cost (INR)']}, "
        f"Marketing Spend: {row['Marketing Spend (INR)']}"
        for _, row in df.iterrows()
    ]

    # Create embeddings
    doc_embeddings = model.encode(docs)

    def answer_question(query: str):
        # Encode query and compute similarity
        query_emb = model.encode([query])
        sims = cosine_similarity(query_emb, doc_embeddings)[0]

        # Pick the most relevant entries
        top_indices = sims.argsort()[-3:][::-1]
        context = "\n".join([docs[i] for i in top_indices])

        # Generate a concise answer (simple rule-based summary)
        answer = f"Based on the top relevant data:\n{context}\n\nThis suggests that your query '{query}' relates most to these months or figures."
        return answer

    return answer_question


# Initialize agent globally (so it’s not reloaded each time)
_agent = build_agent()

def query_agent(query: str):
    """Used by Streamlit UI to answer business queries"""
    return _agent(query)
