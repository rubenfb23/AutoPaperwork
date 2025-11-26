from .DocumentLoaderBase import DocumentLoaderBase
from langchain_community.document_loaders import JSONLoader


class JSONDocumentLoader(DocumentLoaderBase):
    """
    Concrete implementation of a document loader for JSON files.

    Args:
        json_path (str): The path to the JSON file.

    Attributes:
        json_path (str): The path to the JSON file.

    Methods:
        load: Loads the JSON file and returns the loaded documents.
    """

    def __init__(self, json_path):
        self.json_path = json_path

    def load(self):
        """
        Loads the JSON file and returns the loaded documents.

        Returns:
            list: The loaded documents.
        """
        loader = JSONLoader(
            file_path=self.json_path,
            jq_schema='.messages[].content',
            text_content=False)
        docs = loader.load()
        print("Loaded JSON")
        return docs
  