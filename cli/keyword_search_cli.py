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

class CacheData:
    def __init__(self):
        Path('cache').mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.docmap = {}
        self.term_frequencies = Counter()
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
        for token in tokens:
            if self.cache_data.index.get(token):
                self.cache_data.index[token].add(doc_id)
            else:
                self.cache_data.index[token] = set()
                self.cache_data.index[token].add(doc_id)
            self.cache_data.term_frequencies[token] += 1

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
        except Exception as e:
            print(e)
            exit(1)
    

    def get_tf(self, doc_id, term):
        self.load()
        tokenized_term = self.tokenizate_term(term)

        tokenized_text = self.tokenizate_text(self.cache_data.docmap.get(int(doc_id), ""))

        counter = Counter(tokenized_text)
        result = counter[tokenized_term]
        # print(result)
        return result

    def idf(self, term):
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
        idf_r = math.log((n + 1) / (count + 1))
        # print(f"Inverse document frequency of '{term}': {idf_r:.2f}")
        return idf_r


    def tfidf(self, id, term):
        self.load()
        tf = self.get_tf(id, term)
        idf_r = self.idf(term)
        tf_idf = tf * idf_r
        print(f"TF-IDF score of '{term}' in document '{id}': {tf_idf:.2f}")



def main() -> None:
    parser, args = create_parser()
    cache_data = CacheData()
    invert = None
    if args.command == "build":
        invert = InvertedIndex(args, cache_data)
        invert.build()
    elif args.command == "search":
        search = KeywordSearch(args, cache_data)
        search.search_for_query(args.query)
    elif args.command == "tf":
        if not invert:
            invert = InvertedIndex(args, cache_data)
        invert.get_tf(args.id, args.term)
    elif args.command == "idf":
        if not invert:
            invert = InvertedIndex(args, cache_data)
        invert.idf(args.term)
    elif args.command == "tfidf":
        if not invert:
            invert = InvertedIndex(args, cache_data)
        invert.tfidf(args.id, args.term)




if __name__ == "__main__":
    main()
