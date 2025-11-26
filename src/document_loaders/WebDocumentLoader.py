from .DocumentLoaderBase import DocumentLoaderBase
from langchain_community.document_loaders import WebBaseLoader


class WebDocumentLoader(DocumentLoaderBase):
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
