import argparse
import json
import re
import string
from nltk.stem import PorterStemmer



class LoadingFile():
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loaded_data = self.load_file()
        self.stop_words = self.get_stop_words()


    def get_stop_words(self):
        try:
            with open('data/stopwords.txt', 'r') as f:
                f_data = f.read()
                return set(f_data.splitlines())
        except Exception as e:
            print(e)
            exit(1)


    def load_file(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except Exception as e :
            print(e)
            exit(1)


def CreateParser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    parser.add_argument('--data', default='data/movies.json')
    first_subparser = parser.add_subparsers(dest="command")

    search =  first_subparser.add_parser('search')
    search.add_argument('query')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)
    else:
        return (parser, args)



class KeywordSearch():
    def __init__(self, args):
        self.loaded_file = LoadingFile(args.data)
        self.stemmer = PorterStemmer()


    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text


    def tokenizate_text(self, text: str) -> str:
        text = KeywordSearch.preprocess_text(text)
        tokenizated_text = [self.stemmer.stem(token) for token in text.split()]
        valid_tokens = []
        for token in tokenizated_text:
            if token and token not in self.loaded_file.stop_words:
                valid_tokens.append(token)
        return valid_tokens


    def search_for_query(self, query):
        result = []
        to_list = self.loaded_file.loaded_data['movies']
        token_query = self.tokenizate_text(query)
        for l in to_list:
            temp_title = self.tokenizate_text(l['title'])
            for token in token_query:
                if any(token in title_token for title_token in temp_title):
                    result.append(l['title'])
                    break


        n = len(result)
        for i , title in zip(range(n), result):
            print(f"{i + 1}. {title}")
        return result

def main() -> None:
    parser, args = CreateParser()

    search = KeywordSearch(args)

    search.search_for_query(args.query)


if __name__ == "__main__":
    main()
