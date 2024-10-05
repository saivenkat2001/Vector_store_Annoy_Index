from old.vector_store import VectorStore
import numpy as np
from PyPDF2 import PdfReader

vector_store = VectorStore()

pdf_reader = PdfReader("Cat.pdf")
sentences = ""
for page in pdf_reader.pages:
    sentences += page.extract_text()

sentences = sentences.split(".")
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

# sentences = [
#     "I eat mango",
#     "mango is my favourite fruit",
#     "mango, apple and oranges are fruits",
#     "fruits are good for health"
# ]

number_of_chunks = 5

chunk_size = len(sentences) // number_of_chunks + (len(sentences) % number_of_chunks > 0)

chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

# Tokenization and building vocabulary for all chunks
vocabulary = set()

for chunk in chunks:
    for sentence in chunk:
        tokens = sentence.lower().split()
        vocabulary.update(tokens)

# Creating a global word to index mapping
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization using the global vocabulary
chunk_id = 0  # ID for the current chunk

for chunk in chunks:
    # Vectorization for the current chunk
    sentence_vectors = {}
    for sentence in chunk:
        tokens = sentence.lower().split()
        vector = np.zeros(len(vocabulary))
        for token in tokens:
            if token in word_to_index:  # Ensure token is in the vocabulary
                vector[word_to_index[token]] += 1  # Count occurrences of each token
        sentence_vectors[sentence] = vector

    # Add vectors to the vector store, using chunk_id
    for sentence, vector in sentence_vectors.items():
        vector_store.add_vectors(chunk_id, sentence, vector)

    chunk_id += 1  # Increment chunk_id for the next chunk

query_sentence = "Cat is a animal"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()

for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

similar_sentences = vector_store.find_similarity_vector(query_vector, num_results=5)

print("Query sentence", query_sentence)
print("Similar sentences: ")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")
