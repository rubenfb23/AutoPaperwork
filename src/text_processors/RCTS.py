import TextProcessor
from langchain_text_splitters import RecursiveCharacterTextSplitter


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