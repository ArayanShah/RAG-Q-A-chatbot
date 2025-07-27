# app.py

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------- Title ----------
st.set_page_config(page_title="Loan Q&A RAG Chatbot", layout="centered")
st.title("💬 Loan Approval Q&A Chatbot (RAG + HuggingFace)")

@st.cache_resource
def load_model_and_data():
    # Load dataset
    df = pd.read_csv("Training Dataset.csv")
    df['Credit_History'] = pd.to_numeric(df['Credit_History'], errors='coerce')
    df['ApplicantIncome'] = pd.to_numeric(df['ApplicantIncome'], errors='coerce')
    df = df.dropna(subset=['Credit_History', 'ApplicantIncome', 'Loan_Status', 'Property_Area'])
    df['Loan_Status'] = df['Loan_Status'].astype(str)
    df['Property_Area'] = df['Property_Area'].astype(str)

    documents = []
    loan_approved = df[df['Loan_Status'] == 'Y']
    loan_rejected = df[df['Loan_Status'] == 'N']
    documents.append(f"The total number of loans that were approved is: {len(loan_approved)}")
    documents.append(f"The total number of loans that were rejected is: {len(loan_rejected)}")

    most_common_credit = loan_approved['Credit_History'].mode()[0]
    documents.append(f"The most common value of the 'Credit_History' column among approved loans is: {most_common_credit}")

    top_area = loan_approved['Property_Area'].mode()[0]
    documents.append(f"The 'Property_Area' that has the most approved loans is: {top_area}")

    for area in df['Property_Area'].unique():
        avg_income = df[df['Property_Area'] == area]['ApplicantIncome'].mean()
        documents.append(f"The average applicant income in the '{area}' area is: {avg_income:.2f}")

    # Embedding
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedder.encode(documents)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))

    # Load FLAN-T5 model
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    return documents, embedder, index, tokenizer, model

# Load everything
documents, embedder, index, tokenizer, model = load_model_and_data()

def retrieve_documents(query, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

def generate_answer(query):
    context = "\n".join(retrieve_documents(query))
    prompt = (
        f"You are a data expert. Use only the information in the DATA section below to answer accurately.\n\n"
        f"=== DATA ===\n{context}\n=== END ===\n\n"
        f"Q: {query}\nA:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- UI ----------
query = st.text_input("🔍 Ask a question about the loan dataset:")
if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)
    st.success("✅ Answer:")
    st.write(answer)
