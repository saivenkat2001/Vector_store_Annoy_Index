import PyPDF2
import numpy as np
from annoy import AnnoyIndex
import pickle
import re
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def create_embeddings(sentences, model):
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    return sentence_embeddings

def create_vector_store(sentences, sentence_embeddings, index_file='annoy_index.ann'):
    vector_size = sentence_embeddings.shape[1] 
    annoy_index = AnnoyIndex(vector_size, metric='angular')
    
    for i, embedding in enumerate(sentence_embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save(index_file)
    with open("vector_data.pkl", "wb") as f:
        pickle.dump((sentences, sentence_embeddings), f)

    print(f"Vector store created and Annoy index saved to {index_file}")

def query_to_embedding(query, model):
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    return query_embedding

def get_top_n_similar_sentences(query_embedding, index_file='annoy_index.ann', n=3):
    with open("vector_data.pkl", "rb") as f:
        sentences, sentence_embeddings = pickle.load(f)
    
    vector_size = sentence_embeddings.shape[1]
    annoy_index = AnnoyIndex(vector_size, metric='angular')
    annoy_index.load(index_file)
    top_n_indices, distances = annoy_index.get_nns_by_vector(query_embedding, n, include_distances=True)
    
    top_n_sentences = [(sentences[i], 1 - dist) for i, dist in zip(top_n_indices, distances)]

    return top_n_sentences

def clean_sentence(sentence):
    cleaned = re.sub(r'\[?\d+\]?', '', sentence)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def main(pdf_path, query):
    print(f"Analyzing PDF: {pdf_path}")
    print(f"Query: {query}\n")

    text = extract_text_from_pdf(pdf_path)
    sentences = split_text_into_sentences(text)
    
    print("Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Creating sentence embeddings...")
    sentence_embeddings = create_embeddings(sentences, model)
    
    print("Creating vector store...")
    create_vector_store(sentences, sentence_embeddings)
    
    print("Processing query...")
    query_embedding = query_to_embedding(query, model)
    
    print("Retrieving similar sentences...")
    top_n = get_top_n_similar_sentences(query_embedding, n=5)
    
    print("\nTop similar sentences and their similarity scores:\n")
    for i, (sentence, score) in enumerate(top_n, 1):
        cleaned_sentence = clean_sentence(sentence)
        print(f"{i}. Score: {score:.4f}")
        print(f"Sentence: {cleaned_sentence}")
        print()

if __name__ == "__main__":
    pdf_path = 'Cat.pdf' 
    query = "How many legs does a cat have?"
    main(pdf_path, query)