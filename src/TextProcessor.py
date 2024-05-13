from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
import json
import requests


class TextProcessor(ABC):
    """
    Abstract base class for text processing.

    This class serves as a base for implementing text processing functionality.
    Subclasses should override the `process` method to define their specific
    text processing logic.

    Attributes:
        None

    Methods:
        process(documents): Processes a list of documents and returns the processed result.
    """

    @abstractmethod
    def process(self, documents):
        """
        Processes a list of documents and returns the processed result.

        Args:
            documents (list): A list of documents to be processed.

        Returns:
            The processed result.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass


class RCTS(TextProcessor):
    """
    Concrete implementation of a text processor that splits text into chunks.

    Args:
        separators (list): List of separators used to split the text.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Number of characters to overlap between adjacent chunks.
        length_function (function): Function to calculate the length of a chunk.
        is_separator_regex (bool, optional): Flag indicating whether the separators are regular expressions.

    Attributes:
        separators (list): List of separators used to split the text.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Number of characters to overlap between adjacent chunks.
        length_function (function): Function to calculate the length of a chunk.
        is_separator_regex (bool): Flag indicating whether the separators are regular expressions.

    Methods:
        process: Splits the input documents into chunks using the specified parameters.

    Returns:
        list: List of processed documents, where each document is a list of chunks.
    """
    def __init__(self, separators, chunk_size, chunk_overlap, length_function, is_separator_regex=False):
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.is_separator_regex = is_separator_regex

    def process(self, documents):
        processed_docs = []
        processed_docs.extend(RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=self.is_separator_regex
        ).split_documents(documents))
        print("Document splitted")
        return processed_docs


class RSJSON(TextProcessor):
    """
    Concrete implementation of a text processor that splits JSON text into chunks.

    Args:
        max_chunk_size (int): The maximum size of each chunk.

    Attributes:
        max_chunk_size (int): The maximum size of each chunk.

    Methods:
        process(documents): Processes the JSON documents and splits them into chunks.

    """

    def __init__(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size

    def process(self, documents):
        """
        Processes the JSON documents and splits them into chunks.

        Args:
            documents (str): The JSON documents to be processed.

        Returns:
            list: A list of processed JSON chunks.

        """
        processed_json = []
        json_data = json.loads(documents)
        for file in json_data:
            processed_json.extend(RecursiveJsonSplitter(
                max_chunk_size=self.max_chunk_size,
            ).split_json(file))
        return processed_json
