import json

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