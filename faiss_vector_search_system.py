import os
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class VectorSearchSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.sentences = []

    def load_sentences(self, file_path=None):
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        else:
            sentences = [
                "Artificial Intelligence is the future",
                "Machine learning enables smart systems",
                "Deep learning uses neural networks",
                "Dogs are friendly animals",
                "Cats are independent pets",
                "Football is a global sport",
                "Cricket is popular in India",
                "NLP helps in text understanding"
            ]
        return sentences

    def build_index(self, sentences):
        self.sentences = sentences
        embeddings = self.model.encode(sentences)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        print(f"\nQuery: {query}")
        for i, idx in enumerate(indices[0]):
            score = 1 / (1 + distances[0][i])
            print(f"{i+1}. {self.sentences[idx]} (Score: {round(score,3)})")

    def save(self, path="faiss_db"):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.sentences, f)

    def load(self, path="faiss_db"):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".pkl", "rb") as f:
            self.sentences = pickle.load(f)


if __name__ == "__main__":
    start = time.time()

    system = VectorSearchSystem()
    sentences = system.load_sentences("data.txt")

    system.build_index(sentences)
    system.save()

    while True:
        query = input("\nEnter query (type 'exit' to stop): ")
        if query.lower() == "exit":
            break
        system.search(query, top_k=3)

    end = time.time()
    print(f"\nExecution Time: {round(end - start, 2)} seconds")