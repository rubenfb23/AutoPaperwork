from .VectorStore import VectorStore
from langchain_community.vectorstores.faiss import FAISS


class FAISSVectorStore(VectorStore):
    """
    A vector store implementation using FAISS.

    Attributes:
        documents (list): The list of documents.
        embeddings (object): The embeddings object used to create the vectors.
    """

    def __init__(self, embeddings):
        """
        Initializes a new instance of the FAISSVectorStore class.

        Args:
            embeddings (object): The embeddings object used to create the vectors.
        """
        self.embeddings = embeddings

    def initialize(self, documents):
        """
        Initializes the vector store using FAISS.

        Args:
            documents (list): The list of documents.

        Returns:
            object: The initialized FAISS object.
        """
        return FAISS.from_documents(documents, self.embeddings)