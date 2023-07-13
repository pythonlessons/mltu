import os
import json
import typing
from tqdm import tqdm

class CustomTokenizer:
    def __init__(
            self, 
            split=" ", 
            lower=True, 
            start_token="<start>", 
            end_token="<eos>",
            # filters="!'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"
            filters = ['!', "'", '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n'],
            filter_nums = True,
        ) -> None:
        self.split = split
        self.lower = lower
        self.index_word = {}
        self.word_index = {}
        self.max_length = 0
        self.start_token = start_token
        self.end_token = end_token
        self.filters = filters
        self.filter_nums = filter_nums
        self.start = 1

    @property
    def start_token_index(self):
        return self.word_index[self.start_token]
    
    @property
    def end_token_index(self):
        return self.word_index[self.end_token]

    def sort(self):
        self.index_word = dict(enumerate(dict(sorted(self.word_index.items())), start=self.start))
        self.word_index = {v: k for k, v in self.index_word.items()}

    def split_line(self, line):
        line = line.lower() if self.lower else line

        # split line with split token and check for filters
        line_tokens = line.split(self.split)

        new_tokens = []

        for index, token in enumerate(line_tokens):
            filtered_tokens = ['']
            for c_index, char in enumerate(token):
                if char in self.filters or (self.filter_nums and char.isdigit()):
                    filtered_tokens += [char, ''] if c_index != len(token) -1 else [char]
                else:
                    filtered_tokens[-1] += char

            new_tokens += filtered_tokens
            if index != len(line_tokens) -1:
                new_tokens += [self.split]


        # char_line = [char for char in line]
        # new = ['']
        # for char in char_line:
        #     if char in self.filters or char == self.split:
        #         new += [char, ''] 
        #     else:
        #         new[-1] += char

        # remove "" tokens
        new_tokens = [token for token in new_tokens if token != '']

        return new_tokens

    def fit_on_texts(self, lines):
        self.word_index = {key: value for value, key in enumerate([self.start_token, self.end_token, self.split] + self.filters)}
        
        for line in tqdm(lines, desc="Fitting tokenizer"):
            line_tokens = self.split_line(line)
            self.max_length = max(self.max_length, len(line_tokens) +2) # +2 for start and end tokens

            for token in line_tokens:
                if token not in self.word_index:
                    self.word_index[token] = len(self.word_index)

        self.sort()

    def update(self, lines):
        new_tokens = 0
        for line in tqdm(lines, desc="Updating tokenizer"):
            line_tokens = self.split_line(line)
            self.max_length = max(self.max_length, len(line_tokens) +2) # +2 for start and end tokens
            for token in line_tokens:
                if token not in self.word_index:
                    self.word_index[token] = len(self.word_index)
                    new_tokens += 1

        self.sort()
        print(f"Added {new_tokens} new tokens")

    def detokenize(self, sequences, remove_start_end=True):
        lines = []
        for sequence in sequences:
            line = ""
            for token in sequence:
                if token == 0:
                    break
                if remove_start_end and (token == self.start_token_index or token == self.end_token_index):
                    continue

                line += self.index_word[token]

            lines.append(line)

        return lines

    def texts_to_sequences(self, lines, include_start_end=True):
        sequences = []
        for line in lines:
            line_tokens = self.split_line(line)
            sequence = [self.word_index[word] for word in line_tokens if word in self.word_index]
            if include_start_end:
                sequence = [self.word_index[self.start_token]] + sequence + [self.word_index[self.end_token]]

            sequences.append(sequence)

        return sequences
    
    def save(self, path, type="json"):
        serialised_dict = self.dict()
        if type == "json":
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(serialised_dict, f)

    def dict(self):
        return {
            "split": self.split,
            "lower": self.lower,
            "index_word": self.index_word,
            "max_length": self.max_length,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "filters": self.filters,
            "filter_nums": self.filter_nums,
            "start": self.start
        }

    @staticmethod
    def load(path: typing.Union[str, dict], type="json"):
        if isinstance(path, str):
            if type == "json":
                with open(path, "r") as f:
                    load_dict = json.load(f)

        elif isinstance(path, dict):
            load_dict = path

        tokenizer = CustomTokenizer()
        tokenizer.split = load_dict["split"]
        tokenizer.lower = load_dict["lower"]
        tokenizer.index_word = {int(k): v for k, v in load_dict["index_word"].items()}
        tokenizer.max_length = load_dict["max_length"]
        tokenizer.start_token = load_dict["start_token"]
        tokenizer.end_token = load_dict["end_token"]
        tokenizer.filters = load_dict["filters"]
        tokenizer.filter_nums = bool(load_dict["filter_nums"])
        tokenizer.start = load_dict["start"]
        tokenizer.word_index = {v: int(k) for k, v in tokenizer.index_word.items()}

        return tokenizer
    
    @property
    def lenght(self):
        return len(self.index_word)

    def __len__(self):
        return len(self.index_word)