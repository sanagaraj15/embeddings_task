import os
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class DataLoader:
    @staticmethod
    def load_sentences(file_path=None):
        if file_path and os.path.exists(file_path):
            print("[INFO] Loading sentences from file...")
            with open(file_path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        else:
            print("[INFO] Using default dataset...")
            sentences = [
                "Artificial Intelligence is transforming industries",
                "Machine learning enables computers to learn from data",
                "Deep learning uses neural networks",
                "Dogs are loyal animals",
                "Cats are independent pets",
                "Football is a popular sport",
                "Cricket is loved in India",
                "Natural Language Processing works with text"
            ]
        return sentences

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=4):
        print("[INFO] Loading model...")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def generate(self, sentences):
        print("[INFO] Generating embeddings...")
        all_embeddings = []

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            print(f"[INFO] Processing batch {i//self.batch_size + 1}")
            emb = self.model.encode(batch)
            all_embeddings.extend(emb)

        return np.array(all_embeddings)

class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.sentences = []

    def add(self, embeddings, sentences):
        self.index.add(np.array(embeddings))
        self.sentences.extend(sentences)
        print("[INFO] Data added to FAISS index")

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            score = 1 / (1 + dist)  # normalize score
            results.append((self.sentences[idx], score))

        return results

    def save(self, path="semantic_db"):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.sentences, f)
        print("[INFO] Database saved")

    def load(self, path="semantic_db"):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".pkl", "rb") as f:
            self.sentences = pickle.load(f)
        print("[INFO] Database loaded")



class SemanticSearchSystem:
    def __init__(self):
        self.generator = EmbeddingGenerator()
        self.db = None

    def build(self, sentences):
        embeddings = self.generator.generate(sentences)
        dim = embeddings.shape[1]
        self.db = VectorDB(dim)
        self.db.add(embeddings, sentences)

    def query(self, text, top_k=3):
        query_embedding = self.generator.model.encode([text])
        results = self.db.search(query_embedding, top_k)

        print(f"\n Query: {text}")
        for i, (sentence, score) in enumerate(results):
            print(f"{i+1}. {sentence}  (Score: {round(score,3)})")

if __name__ == "__main__":
    start = time.time()
    sentences = DataLoader.load_sentences("data.txt")
    system = SemanticSearchSystem()
    system.build(sentences)
    system.db.save("semantic_db")
    while True:
        query = input("\nEnter your query (type 'exit' to stop): ")
        if query.lower() == "exit":
            break
        system.query(query, top_k=3)

    end = time.time()
    print(f"\n[INFO] Execution Time: {round(end - start, 2)} seconds")