import document_loaders.DocumentLoaderBase as DocumentLoaderBase
from langchain_community.document_loaders import DirectoryLoader


class CompleteDirectoryLoader(DocumentLoaderBase):
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
        """
        Loads the documents from the specified directory and returns them.

        Returns:
            list: A list of loaded documents.

        Example:
            loader = CompleteDirectoryLoader('/path/to/directory')
            documents = loader.load()
        """
        loader = DirectoryLoader(self.directory_path)
        docs = loader.load()
        print("Loaded directory")
        return docs
