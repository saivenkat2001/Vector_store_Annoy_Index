import numpy as np
import os
import pickle

class VectorStore():
    def __init__(self, similarity_type="cosine", persist_directory="./cat"):
        self.vector_data = {} # to store vectors
        self.vector_index = {} # indexing the structure for retrival
        self.similarity_type = similarity_type
        self.persist_directory = persist_directory
        
        os.makedirs(self.persist_directory, exist_ok=True)

        self.load_vectors()

    def save_vectors(self):
        # Save each chunk's vectors in separate files
        for chunk_id, vectors in self.vector_data.items():
            with open(os.path.join(self.persist_directory, f'vectors_{chunk_id}.pkl'), 'wb') as f:
                pickle.dump(vectors, f)

    def load_vectors(self):
        # Load vectors from chunk files
        for filename in os.listdir(self.persist_directory):
            if filename.startswith("vectors_") and filename.endswith(".pkl"):
                chunk_id = filename.split("_")[1].split(".")[0]
                with open(os.path.join(self.persist_directory, filename), 'rb') as f:
                    vectors = pickle.load(f)
                    self.vector_data[chunk_id] = vectors
                # Rebuild the index for this chunk
                for vector_id, vector in vectors.items():
                    self._update_index(vector_id, vector)

    def add_vectors(self, chunk_id, vector_id, vector):
        # Store vectors in the specific chunk
        if chunk_id not in self.vector_data:
            self.vector_data[chunk_id] = {}
        self.vector_data[chunk_id][vector_id] = vector
        self._update_index(vector_id, vector)
        self.save_vectors()

    def get_vector(self, vector_id):
        # Return vector from any chunk
        for vectors in self.vector_data.values():
            if vector_id in vectors:
                return vectors[vector_id]
        return None

    def _update_index(self, vector_id, vector):
        # Update the similarity index for the provided vector
        for chunk_id, vectors in self.vector_data.items():
            for existing_id, existing_vector in vectors.items():
                similarity = self._calculate_similarity(vector, existing_vector)
                if chunk_id not in self.vector_index:
                    self.vector_index[chunk_id] = {}
                self.vector_index[chunk_id][existing_id] = similarity

    def _calculate_similarity(self, vector1, vector2):
        if self.similarity_type == "cosine":
            return self._cosine_similarity(vector1, vector2)
        elif self.similarity_type == "euclidean":
            return self._euclidean_similarity(vector1, vector2)
        elif self.similarity_type == "jaccard":
            return self._jaccard_similarity(vector1, vector2)
        elif self.similarity_type == "manhattan":
            return self._manhattan_similarity(vector1, vector2)
        elif self.similarity_type == "hamming":
            return self._hamming_similarity(vector1, vector2)
        elif self.similarity_type == "dot":
            return self._dot_product(vector1, vector2)
        elif self.similarity_type == "angular":
            return self._angular_similarity(vector1, vector2)
        else:
            raise ValueError(f"Unknown similarity type: '{self.similarity_type}'")

    def _cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norm_product == 0:
            return 0
        return dot_product / norm_product

    def _euclidean_similarity(self, vector1, vector2):
        distance = np.linalg.norm(vector1 - vector2)
        return 1 / (1 + distance)

    def _jaccard_similarity(self, vector1, vector2):
        intersection = np.sum(np.minimum(vector1, vector2))
        union = np.sum(np.maximum(vector1, vector2))
        if union == 0:
            return 0
        return intersection / union

    def _manhattan_similarity(self, vector1, vector2):
        # Manhattan distance: sum of absolute differences
        distance = np.sum(np.abs(vector1 - vector2))
        return 1 / (1 + distance)

    def _hamming_similarity(self, vector1, vector2):
        # Hamming distance: number of differing elements
        distance = np.sum(vector1 != vector2)
        return 1 / (1 + distance)

    def _dot_product(self, vector1, vector2):
        # Dot product similarity: sum of element-wise products
        return np.dot(vector1, vector2)

    def _angular_similarity(self, vector1, vector2):
        # Angular similarity: angle between vectors (cosine similarity's inverse)
        cosine_sim = self._cosine_similarity(vector1, vector2)
        return 1 - (np.arccos(np.clip(cosine_sim, -1, 1)) / np.pi)

    def find_similarity_vector(self, query_vector, num_results=5):
        results = []
        # Iterate over each chunk in vector_data
        for chunk_id, vectors in self.vector_data.items():
            for vector_id, vector in vectors.items():
                similarity = self._calculate_similarity(query_vector, vector)
                results.append((vector_id, similarity))

        # Sort results by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results (highest similarity)
        return results[:num_results]