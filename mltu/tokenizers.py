import os
import json
import typing
from tqdm import tqdm

class CustomTokenizer:
    """ Custom Tokenizer class to tokenize and detokenize text data into sequences of integers

    Args:
        split (str, optional): Split token to use when tokenizing text. Defaults to " ".
        char_level (bool, optional): Whether to tokenize at character level. Defaults to False.
        lower (bool, optional): Whether to convert text to lowercase. Defaults to True.
        start_token (str, optional): Start token to use when tokenizing text. Defaults to "<start>".
        end_token (str, optional): End token to use when tokenizing text. Defaults to "<eos>".
        filters (list, optional): List of characters to filter out. Defaults to 
            ['!', "'", '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', 
            '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n'].
        filter_nums (bool, optional): Whether to filter out numbers. Defaults to True.
        start (int, optional): Index to start tokenizing from. Defaults to 1.
    """
    def __init__(
            self, 
            split: str=" ", 
            char_level: bool=False,
            lower: bool=True, 
            start_token: str="<start>", 
            end_token: str="<eos>",
            filters: list = ['!', "'", '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n'],
            filter_nums: bool = True,
            start: int=1,
        ) -> None:
        self.split = split
        self.char_level = char_level
        self.lower = lower
        self.index_word = {}
        self.word_index = {}
        self.max_length = 0
        self.start_token = start_token
        self.end_token = end_token
        self.filters = filters
        self.filter_nums = filter_nums
        self.start = start

    @property
    def start_token_index(self):
        return self.word_index[self.start_token]
    
    @property
    def end_token_index(self):
        return self.word_index[self.end_token]

    def sort(self):
        """ Sorts the word_index and index_word dictionaries"""
        self.index_word = dict(enumerate(dict(sorted(self.word_index.items())), start=self.start))
        self.word_index = {v: k for k, v in self.index_word.items()}

    def split_line(self, line: str):
        """ Splits a line of text into tokens

        Args:
            line (str): Line of text to split

        Returns:
            list: List of string tokens
        """
        line = line.lower() if self.lower else line

        if self.char_level:
            return [char for char in line]

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

        new_tokens = [token for token in new_tokens if token != '']

        return new_tokens

    def fit_on_texts(self, lines: typing.List[str]):
        """ Fits the tokenizer on a list of lines of text
        This function will update the word_index and index_word dictionaries and set the max_length attribute

        Args:
            lines (typing.List[str]): List of lines of text to fit the tokenizer on
        """
        self.word_index = {key: value for value, key in enumerate([self.start_token, self.end_token, self.split] + self.filters)}
        
        for line in tqdm(lines, desc="Fitting tokenizer"):
            line_tokens = self.split_line(line)
            self.max_length = max(self.max_length, len(line_tokens) +2) # +2 for start and end tokens

            for token in line_tokens:
                if token not in self.word_index:
                    self.word_index[token] = len(self.word_index)

        self.sort()

    def update(self, lines: typing.List[str]):
        """ Updates the tokenizer with new lines of text
        This function will update the word_index and index_word dictionaries and set the max_length attribute

        Args:
            lines (typing.List[str]): List of lines of text to update the tokenizer with
        """
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

    def detokenize(self, sequences: typing.List[int], remove_start_end: bool=True):
        """ Converts a list of sequences of tokens back into text

        Args:
            sequences (typing.list[int]): List of sequences of tokens to convert back into text
            remove_start_end (bool, optional): Whether to remove the start and end tokens. Defaults to True.
        
        Returns:
            typing.List[str]: List of strings of the converted sequences
        """
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

    def texts_to_sequences(self, lines: typing.List[str], include_start_end: bool=True):
        """ Converts a list of lines of text into a list of sequences of tokens
        
        Args:
            lines (typing.list[str]): List of lines of text to convert into tokenized sequences
            include_start_end (bool, optional): Whether to include the start and end tokens. Defaults to True.

        Returns:
            typing.List[typing.List[int]]: List of sequences of tokens
        """
        sequences = []
        for line in lines:
            line_tokens = self.split_line(line)
            sequence = [self.word_index[word] for word in line_tokens if word in self.word_index]
            if include_start_end:
                sequence = [self.word_index[self.start_token]] + sequence + [self.word_index[self.end_token]]

            sequences.append(sequence)

        return sequences
    
    def save(self, path: str, type: str="json"):
        """ Saves the tokenizer to a file
        
        Args:
            path (str): Path to save the tokenizer to
            type (str, optional): Type of file to save the tokenizer to. Defaults to "json".
        """
        serialised_dict = self.dict()
        if type == "json":
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(serialised_dict, f)

    def dict(self):
        """ Returns a dictionary of the tokenizer

        Returns:
            dict: Dictionary of the tokenizer
        """
        return {
            "split": self.split,
            "lower": self.lower,
            "char_level": self.char_level,
            "index_word": self.index_word,
            "max_length": self.max_length,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "filters": self.filters,
            "filter_nums": self.filter_nums,
            "start": self.start
        }

    @staticmethod
    def load(path: typing.Union[str, dict], type: str="json"):
        """ Loads a tokenizer from a file

        Args:
            path (typing.Union[str, dict]): Path to load the tokenizer from or a dictionary of the tokenizer
            type (str, optional): Type of file to load the tokenizer from. Defaults to "json".

        Returns:
            CustomTokenizer: Loaded tokenizer
        """
        if isinstance(path, str):
            if type == "json":
                with open(path, "r") as f:
                    load_dict = json.load(f)

        elif isinstance(path, dict):
            load_dict = path

        tokenizer = CustomTokenizer()
        tokenizer.split = load_dict["split"]
        tokenizer.lower = load_dict["lower"]
        tokenizer.char_level = load_dict["char_level"]
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