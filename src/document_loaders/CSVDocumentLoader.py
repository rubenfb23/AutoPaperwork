from .DocumentLoaderBase import DocumentLoaderBase
from langchain_community.document_loaders import CSVLoader


class CSVDocumentLoader(DocumentLoaderBase):
    """
    Concrete implementation of a document loader for CSV files.
    """

    def __init__(self, csv_path):
        """
        Initializes a new instance of the CSVDocumentLoader class.

        Args:
            csv_path (str): The path to the CSV file.
        """
        self.csv_path = csv_path

    def load(self):
        """
        Loads the CSV document.

        Returns:
            list: A list of loaded documents.
        """
        loader = CSVLoader(self.csv_path)
        docs = loader.load()
        print("Loaded CSV")
        return docs
