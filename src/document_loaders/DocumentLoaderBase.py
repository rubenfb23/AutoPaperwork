from abc import ABC, abstractmethod


class DocumentLoaderBase(ABC):
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
