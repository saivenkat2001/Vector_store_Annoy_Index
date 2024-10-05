import PyPDF2
import numpy as np
from annoy import AnnoyIndex
import pickle
from gensim.models import Word2Vec
import re

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

def preprocess_sentences(sentences):
    return [sentence.lower().split() for sentence in sentences]

def create_word2vec_model(sentences):
    return Word2Vec(sentences, vector_size=100, min_count=1, window=5)

def create_embeddings(sentences, model):
    sentence_embeddings = []
    for sentence in sentences:
        if sentence:
            sentence_vec = np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
        else:
            sentence_vec = np.zeros(model.vector_size)
        sentence_embeddings.append(sentence_vec)
    return sentence_embeddings

def create_vector_store(sentences, sentence_embeddings, index_file='annoy_index.ann'):
    vector_size = len(sentence_embeddings[0])
    annoy_index = AnnoyIndex(vector_size, metric='angular')
    
    for i, embedding in enumerate(sentence_embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save(index_file)
    with open("vector_data.pkl", "wb") as f:
        pickle.dump((sentences, sentence_embeddings), f)

    print(f"Vector store created and Annoy index saved to {index_file}")

def query_to_embedding(query, model):
    tokens = query.lower().split()
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if word_vectors:
        query_embedding = np.mean(word_vectors, axis=0) 
    else:
        query_embedding = np.zeros(model.vector_size)  
    return query_embedding

def get_top_n_similar_sentences(query_embedding, index_file='annoy_index.ann', n=5, max_length=30):
    with open("vector_data.pkl", "rb") as f:
        sentences, sentence_embeddings = pickle.load(f)
    
    vector_size = len(sentence_embeddings[0])
    annoy_index = AnnoyIndex(vector_size, metric='angular')
    annoy_index.load(index_file)
    top_n_indices, distances = annoy_index.get_nns_by_vector(query_embedding, n, include_distances=True)
    
    top_n_sentences = [(sentences[i], 1 - dist) for i, dist in zip(top_n_indices, distances)]

    return top_n_sentences

def main(pdf_path, query):
    text = extract_text_from_pdf(pdf_path)
    sentences = split_text_into_sentences(text)
    preprocessed_sentences = preprocess_sentences(sentences)
    model = create_word2vec_model(preprocessed_sentences)
    sentence_embeddings = create_embeddings(preprocessed_sentences, model)
    create_vector_store(sentences, sentence_embeddings)
    query_embedding = query_to_embedding(query, model)
    top_n = get_top_n_similar_sentences(query_embedding, n=5, max_length=30)
    print("\nTop similar sentences and their similarity scores:\n")
    for sentence, score in top_n:
        print(f"Score: {score:.4f}, Sentence: {sentence}")

pdf_path = 'Cat.pdf' 
query = "Cat is an animal"
main(pdf_path, query)