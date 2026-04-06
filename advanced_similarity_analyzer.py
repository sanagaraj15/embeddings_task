import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
                "Artificial Intelligence is powerful",
                "AI is transforming the world",
                "Machine learning is part of AI",
                "Dogs are loyal animals",
                "Cats are cute pets",
                "Football is a popular sport",
                "Cricket is loved in India",
                "Natural Language Processing deals with text"
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
            print(f"[INFO] Batch {i//self.batch_size + 1}")
            emb = self.model.encode(batch)
            all_embeddings.extend(emb)
        return np.array(all_embeddings)

class SimilarityAnalyzer:
    @staticmethod
    def compute_matrix(embeddings):
        print("[INFO] Computing similarity matrix...")
        return cosine_similarity(embeddings, embeddings)
    @staticmethod
    def display_matrix(matrix, sentences):
        print("\n[INFO] Similarity Matrix:\n")
        for i, row in enumerate(matrix):
            row_display = ["{:.2f}".format(score) for score in row]
            print(f"{i}: {row_display} -> {sentences[i]}")
    @staticmethod
    def top_k_pairs(matrix, sentences, k=5):
        print(f"\n[INFO] Top {k} Similar Sentence Pairs:\n")
        pairs = []
        n = len(sentences)

        for i in range(n):
            for j in range(i+1, n):
                pairs.append((matrix[i][j], sentences[i], sentences[j]))

        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

        for i in range(min(k, len(pairs))):
            score, s1, s2 = pairs[i]
            print(f"{i+1}. ({score:.3f})")
            print(f"   - {s1}")
            print(f"   - {s2}")

    @staticmethod
    def filter_threshold(matrix, sentences, threshold=0.7):
        print(f"\n[INFO] Pairs above threshold {threshold}:\n")

        n = len(sentences)
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i][j] >= threshold:
                    print(f"({matrix[i][j]:.3f}) {sentences[i]}  <-->  {sentences[j]}")

class ResultStorage:
    @staticmethod
    def save(matrix, sentences, path="similarity_results.json"):
        print("[INFO] Saving results...")
        data = {
            "sentences": sentences,
            "similarity_matrix": matrix.tolist()
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print("[INFO] Results saved!")

    @staticmethod
    def load(path="similarity_results.json"):
        if not os.path.exists(path):
            raise FileNotFoundError("File not found!")

        print("[INFO] Loading results...")
        with open(path, "r") as f:
            data = json.load(f)

        return np.array(data["similarity_matrix"]), data["sentences"]

if __name__ == "__main__":

    start = time.time()
    sentences = DataLoader.load_sentences("data.txt")
    generator = EmbeddingGenerator(batch_size=3)
    embeddings = generator.generate(sentences)
    analyzer = SimilarityAnalyzer()
    matrix = analyzer.compute_matrix(embeddings)
    analyzer.display_matrix(matrix, sentences)
    analyzer.top_k_pairs(matrix, sentences, k=5)
    analyzer.filter_threshold(matrix, sentences, threshold=0.75)
    ResultStorage.save(matrix, sentences)
    loaded_matrix, loaded_sentences = ResultStorage.load()
    print("\n[INFO] Verification:")
    print("Matrix shape:", loaded_matrix.shape)
    end = time.time()
    print(f"\n[INFO] Execution Time: {round(end - start, 2)} seconds")