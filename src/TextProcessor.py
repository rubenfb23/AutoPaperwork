from abc import ABC, abstractmethod


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    """
    @abstractmethod
    def load(self):
        """
        Loads documents and returns them.
        Must be implemented by any subclass.
        """
        pass


class WebDocumentLoader(DocumentLoader):
    """
    Concrete implementation of a document loader for web documents.
    """
    def __init__(self, website_url):
        self.website_url = website_url

    def load(self):
        loader = WebBaseLoader(self.website_url)
        docs = loader.load()
        print("Loaded website")
        return docs


class CSVDocumentLoader(DocumentLoader):
    """
    Concrete implementation of a document loader for CSV files.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        loader = CSVLoader(self.csv_path)
        docs = loader.load()
        print("Loaded CSV")
        return docs
