from abc import ABC, abstractmethod


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
