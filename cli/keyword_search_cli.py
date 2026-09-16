from parser import create_parser
from file_loader import LoadingFile
import re
import string
from nltk.stem import PorterStemmer
from functools import lru_cache
import pickle
from pathlib import Path

class CacheData:
    def __init__(self):
        Path('cache').mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.docmap = {}
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

    def load(self):
        try:
            with open(self.cache_data.saved_index, 'rb') as f:
                self.cache_data.index = pickle.load(f)
            with open(self.cache_data.saved_docmap, 'rb') as f:
                    self.cache_data.docmap = pickle.load(f)
        except Exception as e:
            print(e)
            exit(1)

    def search_for_query(self, query):
        # if not self.cache_data.index or not self.cache_data.docmap: 
        self.load()
        to_list = self.loaded_file.loaded_data['movies']
        token_query = self.tokenizate_text(query)
        result = [list(self.cache_data.index[token]) for token in token_query][0]


        # for l in to_list:
        #     temp_title = self.tokenizate_text(l['title'])
        #     for token in token_query:
        #         if any(token in title_token for title_token in temp_title):
        #             result.append(l['title'])
        #             break
        # print(len(self.loaded_file.loaded_data['movies']))
        for id in result:
            title = list(self.loaded_file.loaded_data['movies'])[id]['title']
            print(f"{id}. {title}")
        # return result


class InvertedIndex(KeywordSearch):
    def __init__(self, args, cache_data):
        super().__init__(args, cache_data)
        self.file_path = args.data
        self.loaded_file = None
    
    def __add_document(self, doc_id, text):
        tokens = self.tokenizate_text(text)
        for token in tokens:
            if self.cache_data.index.get(token):
                self.cache_data.index[token].add(doc_id)
            else:
                self.cache_data.index[token] = set()
                self.cache_data.index[token].add(doc_id)

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
        except Exception as e:
            print(e)
            exit(1)




def main() -> None:
    parser, args = create_parser()
    cache_data = CacheData()
    if args.command == "build":
        invert = InvertedIndex(args, cache_data)
        invert.build()
    elif args.command == "search":
        search = KeywordSearch(args, cache_data)
        search.search_for_query(args.query)



if __name__ == "__main__":
    main()
