import argparse
import json
import re
from typing import List, TypedDict

import numpy as np
from semantic_parser import create_parser
from sentence_transformers import SentenceTransformer


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class SemanticSearch:
    def __init__(self, model_name, data_path):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
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
            self.embeddings = np.load(self.embeddings_path)
        except Exception:
            created_again = True
            self.build_embeddings(documents)
        if len(self.embeddings) == len(documents):
            return self.embeddings

        if not created_again:
            self.build_embeddings(documents)

        return self.embeddings

    def search(self, query, limit):
        try:
            self.embeddings = np.load(self.embeddings_path)
        except Exception:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        embeded_query = self.generate_embedding(query)
        n = len(self.embeddings)
        score = [
            (cosine_similarity(embeded_query, doc), self.documents[i])
            for i, doc in zip(range(n), self.embeddings)
        ]
        score = sorted(score, key=lambda x: x[0], reverse=True)
        return [
            {
                "score": s[0],
                "title": s[1]["title"],
                "description": s[1]["description"],
            }
            for i, s in zip(range(limit), score[:limit])
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name, data_path):
        super().__init__(model_name, data_path)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.embedded_chunk_path = "cache/chunk_embeddings.npy"
        self.chunk_metadata_path = "cache/chunk_metadata.json"

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        chunks: List[str] = []
        chunk_metadata: List[ChunkMetadata] = []

        for id, doc in self.document_map.items():
            if doc["description"].strip() == "":
                continue
            temp_chunks = semantic_chunk_text(doc["description"], 4, 1)
            chunks.extend(temp_chunks)
            for i, chunk in zip(range(len(temp_chunks)), temp_chunks):
                meta_data = ChunkMetadata(
                    movie_idx=id, chunk_idx=i, total_chunks=len(temp_chunks)
                )
                chunk_metadata.append(meta_data)
        self.chunk_metadata = chunk_metadata
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        try:
            np.save(self.embedded_chunk_path, self.chunk_embeddings, allow_pickle=True)
            with open(self.chunk_metadata_path, "w") as f:
                json.dump(
                    {"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2
                )
        except Exception as e:
            print(e)
            exit(1)
        print(len(chunks))
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        try:
            self.chunk_embeddings = np.load(self.embedded_chunk_path)
            with open(self.chunk_metadata_path) as f:
                self.chunk_metadata = json.load(f)
        except Exception as e:
            self.build_chunk_embeddings(documents)

        return self.chunk_embeddings


def search(query, limit):
    semantic = SemanticSearch("all-MiniLM-L6-v2", "data/movies.json")
    try:
        with open(semantic.data_path) as file:
            semantic.documents = json.load(file)["movies"]
    except Exception as e:
        print(e)
        exit(1)
    semantic.load_or_create_embeddings(semantic.documents)
    limit_search = semantic.search(query, limit)
    for i, s in zip(range(limit), limit_search):
        print(f"{i + 1}. {s['title']} (score: {s['score']:.4f})")
        print(f"{s['description']}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    semantic = SemanticSearch("all-MiniLM-L6-v2", "data/movies.json")
    semantic.embeddings = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {semantic.embeddings[:3]}")
    print(f"Shape: {semantic.embeddings.shape}")


def verify_embeddings():
    semantic = SemanticSearch("all-MiniLM-L6-v2", "data/movies.json")
    try:
        with open("data/movies.json") as file:
            loaded_file = json.load(file)["movies"]
    except Exception as e:
        print(e)
        exit(1)
    semantic.load_or_create_embeddings(loaded_file)
    print(f"Number of docs:   {len(loaded_file)}")
    print(
        f"Embeddings shape: {semantic.embeddings.shape[0]} vectors in {semantic.embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    semantic = SemanticSearch("all-MiniLM-L6-v2", "data/movies.json")
    print(f"Text: {text}")
    embedding = semantic.generate_embedding(text)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    semantic = SemanticSearch("all-MiniLM-L6-v2", "data/movies.json")
    print(f"Model loaded: {semantic.model}")
    print(f"Max sequence length: {semantic.model.max_seq_length}")
    return semantic


def print_chunks(chunked: List[str], l: int, n: int, semantically=False):
    if semantically:
        print(f"Semantically chunking {n} characters")
    else:
        print(f"Chunking {n} characters")
    for i, chunk in zip(range(l), chunked):
        print(f"{i + 1}. {chunk}")


def chunk_text(query: str, chunk_size: int, overlap: int) -> List[str]:
    chunked = []
    splited = query.split()
    l = 0
    n = len(query)

    while l < len(splited):
        temp = " ".join(s for s in splited[l : l + chunk_size])
        l_n = len(splited[l : l + chunk_size])
        if l + l_n < len(splited):
            l += l_n - overlap
        else:
            l += l_n
        chunked.append(temp)

    print_chunks(chunked, l, n)


def semantic_chunk_text(query: str, chunk_size: int, overlap: int) -> List[str]:
    splited = re.split(r"(?<=[.!?])\s+", query)
    chunked = []

    l = 0
    n = len(query)

    while l < len(splited):
        temp = " ".join(s for s in splited[l : l + chunk_size])
        l_n = len(splited[l : l + chunk_size])
        if l + l_n < len(splited):
            l += l_n - overlap
        else:
            l += l_n
        chunked.append(temp)

    # print_chunks(chunked, l, n, semantically=True)
    return chunked


def call_semantic_chunker(model_name, data_path):
    semantic_chunker = ChunkedSemanticSearch(model_name, data_path)
    try:
        with open(semantic_chunker.data_path) as f:
            documents = json.load(f)["movies"]
    except Exception as e:
        print(e)
        exit(1)
    semantic_chunker.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(semantic_chunker.chunk_embeddings)} chunked embeddings")


if __name__ == "__main__":
    parser, args = create_parser()
    if args.command == "verify":
        semantic = verify_model()
    elif args.command == "embed_text":
        embed_text(args.query)
    elif args.command == "verify_embeddings":
        verify_embeddings()
    elif args.command == "embed_query":
        embed_query_text(args.query)
    elif args.command == "search":
        search(args.query, args.limit)
    elif args.command == "chunk":
        chunk_text(args.query, args.chunk_size, args.overlap)
    elif args.command == "semantic_chunk":
        semantic_chunk_text(args.query, args.max_chunk_size, args.overlap)
    elif args.command == "embed_chunks":
        call_semantic_chunker("all-MiniLM-L6-v2", "data/movies.json")
