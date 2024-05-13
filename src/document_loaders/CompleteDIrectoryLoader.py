import DocumentLoader
from langchain_community.document_loaders import DirectoryLoader


class CompleteDirectoryLoader(DocumentLoader):
    """
    Concrete implementation of a document loader for directories.

    This class represents a document loader specifically designed for loading documents from a directory.
    It inherits from the base class DocumentLoader and provides an implementation for the load method.

    Attributes:
        directory_path (str): The path to the directory from which the documents will be loaded.

    Methods:
        load(): Loads the documents from the specified directory and returns them.

    Example usage:
        loader = CompleteDirectoryLoader('/path/to/directory')
        documents = loader.load()
    """
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load(self):
        loader = DirectoryLoader(self.directory_path)
        docs = loader.load()
        print("Loaded directory")
        return docs
    