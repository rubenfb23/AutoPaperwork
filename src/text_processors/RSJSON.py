from .TextProcessor import TextProcessor
import json
from langchain_text_splitters import RecursiveJsonSplitter


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