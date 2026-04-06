import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        return text.strip().lower()
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
                "Artificial Intelligence is revolutionizing industries",
                "Machine learning models improve with data",
                "Deep learning uses neural networks",
                "Birds fly in the sky",
                "Python is widely used for AI development",
                "Natural Language Processing helps understand text",
                "Cricket is popular in India",
                "Dogs are loyal animals"
            ]
        return sentences
class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=4):
        print("[INFO] Loading model...")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
    def generate_embeddings(self, sentences):
        print("[INFO] Generating embeddings in batches...")
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            print(f"[INFO] Processing batch {i//self.batch_size + 1}")
            embeddings = self.model.encode(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)
class EmbeddingAnalyzer:
    @staticmethod
    def show_stats(embeddings):
        print("\n[INFO] Embedding Statistics:")
        print("Shape:", embeddings.shape)
        print("Mean:", np.mean(embeddings))
        print("Std Dev:", np.std(embeddings))
        print("Min:", np.min(embeddings))
        print("Max:", np.max(embeddings))
class EmbeddingStorage:
    @staticmethod
    def save(embeddings, sentences, path="embeddings.json"):
        print("[INFO] Saving embeddings...")
        data = {
            "sentences": sentences,
            "embeddings": embeddings.tolist()
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print("[INFO] Saved successfully!")

    @staticmethod
    def load(path="embeddings.json"):
        if not os.path.exists(path):
            raise FileNotFoundError("Embedding file not found!")

        print("[INFO] Loading embeddings...")
        with open(path, "r") as f:
            data = json.load(f)

        return np.array(data["embeddings"]), data["sentences"]
if __name__ == "__main__":

    start_time = time.time()
    sentences = DataLoader.load_sentences("data.txt")
    sentences = [TextPreprocessor.clean_text(s) for s in sentences]
    generator = EmbeddingGenerator(batch_size=3)
    embeddings = generator.generate_embeddings(sentences)
    EmbeddingAnalyzer.show_stats(embeddings)
    EmbeddingStorage.save(embeddings, sentences)
    loaded_embeddings, loaded_sentences = EmbeddingStorage.load()
    print("\n[INFO] Verification:")
    print("Loaded sentences:", len(loaded_sentences))
    print("Embedding shape:", loaded_embeddings.shape)
    for i in range(min(3, len(sentences))):
        print(f"\nSentence: {loaded_sentences[i]}")
        print("First 5 values:", loaded_embeddings[i][:5])

    end_time = time.time()
    print(f"\n[INFO] Execution Time: {round(end_time - start_time, 2)} seconds")