from abc import ABC, abstractmethod
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    This class serves as a base for implementing document loaders. Subclasses should
    inherit from this class and implement the `load` method to load documents and
    return them.

    Attributes:
        None

    Methods:
        load: Abstract method that must be implemented by subclasses to load documents.

    """

    @abstractmethod
    def load(self):
        """
        Loads documents and returns them.

        This method must be implemented by any subclass of `DocumentLoader`. It should
        handle the logic for loading documents and return them.

        Args:
            None

        Returns:
            documents: The loaded documents.

        """
        pass


class WebDocumentLoader(DocumentLoader):
    """
    Concrete implementation of a document loader for web documents.

    Attributes:
        website_url (str): The URL of the website to load documents from.

    Methods:
        load(): Loads the web documents from the specified website URL.
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
    

class JSONDocumentLoader(DocumentLoader):
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
    

class MarkdownDocumentLoader(DocumentLoader):
    """
    Concrete implementation of a document loader for Markdown files.
    """

    def __init__(self, markdown_path):
        """
        Initializes a new instance of the MarkdownDocumentLoader class.

        Args:
            markdown_path (str): The path to the Markdown file.
        """
        self.markdown_path = markdown_path

    def load(self):
        """
        Loads the Markdown document.

        Returns:
            list: A list of loaded documents.
        """
        loader = UnstructuredMarkdownLoader(self.markdown_path)
        docs = loader.load()
        print("Loaded Markdown")
        return docs
    

class PDFDocumentLoader(DocumentLoader):
    """
    Concrete implementation of a document loader for PDF files.
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load(self):
        """
        Loads the PDF document using PyPDFLoader and returns the loaded document.

        Returns:
            The loaded PDF document.
        """
        loader = PyPDFLoader(self.pdf_path, extract_images=True)
        docs = loader.load()
        print("Loaded PDF")
        return docs