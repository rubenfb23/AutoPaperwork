import DocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader


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