import argparse
import json
from typing import List

import numpy as np
from semantic_parser import create_parser
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings_path = "cache/movie_embeddings.npy"
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError()
        return self.model.encode(text)

    def build_embeddings(self, documents: List[dict]):
        """
        args:
            documents is a list of dictionaries, each representing a movie.
        """
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        string_rep = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(string_rep, show_progress_bar=True)
        try:
            np.save(self.embeddings_path, self.embeddings, allow_pickle=True)
        except Exception as e:
            print(e)
            exit(1)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        created_again = False
        try:
            self.embeddings =  np.load(self.embeddings_path)
        except Exception:
            created_again = True
            self.build_embeddings(documents)
        if length(self.embeddings) == len(documents):
            return self.embeddings

        if not created_again:
            self.build_embeddings(documents)

        return self.embeddings


def verify_embeddings():
    semantic = SemanticSearch()
    try:
        with open('cache/movies.json') as file:
            loaded_file = json.load(file)['movies']
    except Exception as e:
        print(e)
        exit(1)
    semantic.load_or_create_embeddings(loaded_file)
    print(f"Number of docs:   {len(loaded_file)}")
    print(
    f"Embeddings shape: {semantic.embeddings.shape[0]} vectors in {semantic.embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    semantic = SemanticSearch()
    print(f"Text: {text}")
    embedding = semantic.generate_embedding(text)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    semantic = SemanticSearch()
    print(f"Model loaded: {semantic.model}")
    print(f"Max sequence length: {semantic.model.max_seq_length}")
    return semantic


if __name__ == "__main__":
    parser, args = create_parser()
    if args.command == "verify":
        semantic = verify_model()
        try:
            with open("data/movies.json", "r") as file:
                loaded_file = json.load(file)
        except Exception as e:
            print(e)
            exit(1)
        semantic.build_embeddings(loaded_file["movies"][:10])
    elif args.command == "embed_text":
        embed_text(args.query)d
    elif args.command == "verify_embedding":
        verify_embeddings()
