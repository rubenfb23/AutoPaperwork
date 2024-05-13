import DocumentLoader
from langchain_community.document_loaders import PyPDFLoader

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