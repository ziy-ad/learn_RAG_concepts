from parser import create_parser
from file_loader import LoadingFile
import re
import string
from nltk.stem import PorterStemmer
from functools import lru_cache
import pickle
from pathlib import Path
from collections import Counter
import math


BM25_K1 = 1.5
BM25_B = 0.75

class CacheData:
    def __init__(self):
        Path('cache').mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.docmap = {}
        self.doc_lengths = {}
        self.term_frequencies = Counter()
        self.doc_lengths_path = 'cache/doc_lengths.pkl'
        self.saved_frequencies = 'cache/term_frequencies.pkl'
        self.saved_index = 'cache/index.pkl'
        self.saved_docmap = 'cache/docmap.pkl'


class KeywordSearch():
    def __init__(self, args, cache_data):
        self.cache_data = cache_data
        self.loaded_file = LoadingFile(args.data)
        self.stemmer = PorterStemmer()
        self._stem = lru_cache(maxsize=None)(self.stemmer.stem)


    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text


    def tokenizate_text(self, text: str) -> str:
        text = KeywordSearch.preprocess_text(text)
        tokens = (self._stem(t) for t in text.split())
        return [t for t in tokens if t and t not in self.loaded_file.stop_words]

    def tokenizate_term(self, term: str) -> str:
        term = KeywordSearch.preprocess_text(term)
        if len(term.split()) != 1:
            raise ValueError("can only tf one token")
    
        term = self._stem(term)
        return term


    def load(self):
        try:
            with open(self.cache_data.saved_index, 'rb') as f:
                self.cache_data.index = pickle.load(f)
            with open(self.cache_data.saved_docmap, 'rb') as f:
                    self.cache_data.docmap = pickle.load(f)
            with open(self.cache_data.saved_frequencies, 'rb') as f:
                    self.cache_data.term_frequencies = pickle.load(f)
            with open(self.cache_data.doc_lengths_path, 'rb') as f:
                    self.cache_data.doc_lengths = pickle.load(f)
        except Exception as e:
            print(e)
            exit(1)

    def search_for_query(self, query):
        self.load()
        tokenized_text = self.tokenizate_text(query)
        result = []

        for token in tokenized_text:
            extracted_ids = sorted(self.cache_data.index.get(token, set()))

            for id in extracted_ids:
                if id not in result:
                    result.append(id)
                if len(result) >= 5:
                    break
            if len(result) >= 5:
                break

        id_to_title = {m['id']: m['title'] for m in self.loaded_file.loaded_data['movies']}
        for doc_id in result:
            print(f"{doc_id}. {id_to_title.get(doc_id, 'Unknown title')}")
        
        return result

class InvertedIndex(KeywordSearch):
    def __init__(self, args, cache_data):
        super().__init__(args, cache_data)
        self.file_path = args.data
    
    def __add_document(self, doc_id, text):
        tokens = self.tokenizate_text(text)
        raw_length = len(text.split())
        for token in tokens:
            if self.cache_data.index.get(token):
                self.cache_data.index[token].add(doc_id)
            else:
                self.cache_data.index[token] = set()
                self.cache_data.index[token].add(doc_id)
            self.cache_data.term_frequencies[token] += 1
        self.cache_data.doc_lengths[doc_id] = raw_length

    def get_documents(self, term):
        ids = self.cache_data.index[term]
        if ids:
            return sorted(ids)

    def indexing(self):
        for doc_id, text in self.cache_data.docmap.items():
            self.__add_document(doc_id, text)
    

            
    def build(self):
        """It should call load_movies(), then add each movie to both the index and the docmap.
        Use each movie's existing id as its document ID. """
        self.loaded_file = LoadingFile(self.file_path)
        movies = self.loaded_file.loaded_data['movies']
        
        self.cache_data.docmap = {m['id']: f"{m['title']} {m['description']}" for m in movies}
        self.indexing()
        self.save()


    def save(self):
        try:
            with open(self.cache_data.saved_docmap, 'wb') as file:
                pickle.dump(self.cache_data.docmap, file)
            with open(self.cache_data.saved_index, 'wb') as file:
                pickle.dump(self.cache_data.index, file)
            with open(self.cache_data.saved_frequencies, 'wb') as file:
                pickle.dump(self.cache_data.term_frequencies, file)
            with open(self.cache_data.doc_lengths_path, 'wb') as file:
                    pickle.dump(self.cache_data.doc_lengths, file)
        except Exception as e:
            print(e)
            exit(1)
    

    def get_tf(self, doc_id, term):
        self.load()
        tokenized_term = self.tokenizate_term(term)

        tokenized_text = self.tokenizate_text(self.cache_data.docmap.get(int(doc_id), ""))

        counter = Counter(tokenized_text)
        result = counter[tokenized_term]
        return result

    def idf(self, term, for_bm25=False):
        self.load()
        tokenized_term = self.tokenizate_term(term)
        count = 0
        n = 0
        for id, value in self.cache_data.docmap.items():
            n += 1
            tokenized_text = self.tokenizate_text(value)
            counter = Counter(tokenized_text)
            if counter[tokenized_term] > 0:
                count += 1
        if for_bm25:
            return count
        idf_r = math.log((n + 1) / (count + 1))
        # print(f"Inverse document frequency of '{term}': {idf_r:.2f}")
        return idf_r


    def tfidf(self, id, term):
        self.load()
        tf = self.get_tf(id, term)
        idf_r = self.idf(term)
        tf_idf = tf * idf_r
        print(f"TF-IDF score of '{term}' in document '{id}': {tf_idf:.2f}")

    def get_bm25_idf(self, term: str) -> float:
        df = self.idf(term, for_bm25=True)
        n = len(self.cache_data.docmap.keys())
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def bm25_idf_command(self, term):
        self.load()
        score = self.get_bm25_idf(term)
        print(f"BM25 IDF score of '{term}': {score:.2f}")
        return score

    def get_bm25_tf(self, doc_id, term, B=BM25_B, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - B + B * (self.cache_data.doc_lengths.get(int(doc_id), 0) / self.__get_avg_doc_length())
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25_tf_command(self, doc_id, term, *, B=BM25_B, k1=BM25_K1):
        self.load()
        
        bm25_tf = self.get_bm25_tf(doc_id, term,B, k1)
        print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")
        return bm25_tf

    def __get_avg_doc_length(self) -> float:
        if not self.cache_data.doc_lengths:
            return 0.0
        return sum(self.cache_data.doc_lengths.values()) / len(self.cache_data.doc_lengths)
    

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
    

    def bm25_search(self, query, limit):
        self.load()
        tokenized_query = self.tokenizate_text(query)
        id_bm25 = {doc_id: 0 for doc_id in self.cache_data.docmap}

        for doc_id in self.cache_data.docmap:
            for token in tokenized_query:
                id_bm25[doc_id] += self.bm25(doc_id, token)

        ranked = sorted(id_bm25.items(), key=lambda x: x[1], reverse=True)
        print(ranked[:limit])
        return ranked[:limit]

def main() -> None:
    parser, args = create_parser()
    cache_data = CacheData()
    invert = InvertedIndex(args, cache_data)
    if args.command == "build":
        invert.build()
    elif args.command == "search":
        search = KeywordSearch(args, cache_data)
        search.search_for_query(args.query)
    elif args.command == "tf":
        invert.get_tf(args.id, args.term)
    elif args.command == "idf":
        invert.idf(args.term)
    elif args.command == "tfidf":
        invert.tfidf(args.id, args.term)
    elif args.command == "bm25idf":
        invert.bm25_idf_command(args.term)
    elif args.command == "bm25tf":
        invert.bm25_tf_command(args.id, args.term, B=args.b, k1=args.k1)
    elif args.command == "bm25search":
        invert.bm25_search(args.query, args.limit)
    




if __name__ == "__main__":
    main()
