import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class SentenceClusterer:
    def __init__(self, model_name="all-MiniLM-L6-v2", n_clusters=3):
        self.model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters

    def load_sentences(self, file_path=None):
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        else:
            sentences = [
                "Artificial Intelligence is powerful",
                "Machine learning is part of AI",
                "Deep learning uses neural networks",
                "Dogs are loyal animals",
                "Cats are cute pets",
                "Football is a popular sport",
                "Cricket is loved in India",
                "Basketball is a team sport"
            ]
        return sentences

    def generate_embeddings(self, sentences):
        return self.model.encode(sentences)

    def cluster(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels

    def display_clusters(self, sentences, labels):
        clusters = {}
        for sentence, label in zip(sentences, labels):
            clusters.setdefault(label, []).append(sentence)

        for cluster_id, items in clusters.items():
            print(f"\nCluster {cluster_id}:")
            for s in items:
                print(" -", s)

if __name__ == "__main__":
    start = time.time()

    clusterer = SentenceClusterer(n_clusters=3)
    sentences = clusterer.load_sentences("data.txt")
    embeddings = clusterer.generate_embeddings(sentences)
    labels = clusterer.cluster(embeddings)

    print("\nClustering Results:")
    clusterer.display_clusters(sentences, labels)

    end = time.time()
    print(f"\nExecution Time: {round(end - start, 2)} seconds")