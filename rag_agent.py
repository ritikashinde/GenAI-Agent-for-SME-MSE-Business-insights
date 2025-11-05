import os
import pandas as pd
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.docstore.document import Document

# Load model globally
generator = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    device_map="auto"
)

def build_agent(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.getcwd(), "business_data.csv")
    df = pd.read_csv(data_path)

    # Convert rows to text
    text = "\n".join(
        f"Month: {r['Month']}, Sales: {r['Sales (INR)']}, Expenses: {r['Expenses (INR)']}, "
        f"Customers: {r['Customers']}, Inventory Cost: {r['Inventory Cost (INR)']}, "
        f"Marketing Spend: {r['Marketing Spend (INR)']}"
        for _, r in df.iterrows()
    )
    docs = [Document(page_content=text)]

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings & FAISS retriever
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def answer_question(query: str):
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n".join(d.page_content for d in retrieved_docs)

        prompt = (
            f"You are a business analyst. Use the given context to answer clearly and concisely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        response = generator(
            prompt,
            max_new_tokens=150,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.5
        )[0]["generated_text"]

        answer = response.replace(prompt, "").strip()
        return answer

    return answer_question

# Streamlit wrapper
_agent = build_agent()

def query_agent(query: str):
    return _agent(query)
