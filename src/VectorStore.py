from abc import ABC, abstractmethod
from langchain_community.vectorstores.faiss import FAISS


class VectorStore(ABC):
    """
    Abstract base class for vector stores.

    This class serves as a base for implementing vector stores. Subclasses should
    inherit from this class and implement the `initialize` method to initialize the vector store.

    Attributes:
        None

    Methods:
        initialize: Abstract method that must be implemented by subclasses to initialize the vector store.
    """

    @abstractmethod
    def initialize(self):
        """
        Initializes the vector store.

        This method must be implemented by any subclass of `VectorStore`. It should
        handle the logic for initializing the vector store.

        Args:
            self: The instance of the `VectorStore` class.
            
        Returns:
            object: The initialized vector store.

        """
        pass


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