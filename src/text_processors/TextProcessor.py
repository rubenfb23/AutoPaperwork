from abc import ABC, abstractmethod


class TextProcessor(ABC):
    """
    Abstract base class for text processing.

    This class serves as a base for implementing text processing functionality.
    Subclasses should override the `process` method to define their specific
    text processing logic.

    Attributes:
        None

    Methods:
        process(documents): Processes a list of documents and returns the processed result.
    """

    @abstractmethod
    def process(self, documents):
        """
        Processes a list of documents and returns the processed result.

        Args:
            documents (list): A list of documents to be processed.

        Returns:
            The processed result.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass

