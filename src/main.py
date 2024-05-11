import sys
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders.csv_loader import CSVLoader
import sys


def receive_user_input():
    """
    Receives user input from the command line arguments and returns it as a string.

    If command line arguments are provided, it concatenates them into a single string
    and returns it. Otherwise, it returns a default prompt text.

    Returns:
        str: The user input as a string.
    """
    if len(sys.argv) > 1:
        prompt_text = ' '.join(sys.argv[1:])
    else:
        prompt_text = "Cuentame quien eres brevemente y, al acabar, dime *que puedo hacer por ti*"
    return prompt_text


def load_a_website(website_url):
    """
    Loads a website and returns the loaded documents.

    Args:
        website_url (str): The URL of the website to load.

    Returns:
        list: A list of loaded documents.

    """
    loader = WebBaseLoader(website_url)
    docs = loader.load()
    print("Loaded website")
    return docs


def load_a_csv(csv_path):
    """
    Load a CSV file and return the loaded documents.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        list: A list of loaded documents.

    """
    loader = CSVLoader(csv_path)
    docs = loader.load()
    print("Loaded CSV")
    return docs


# Choose the model you want to use for LLM
llm = Ollama(model="llama3:instruct")

output_parser = StrOutputParser()

# Choose the model you want to use for embeddings
embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
            )

# Choose the text splitter you want to use
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,)

# Load the documents from a website
docs = load_a_website("https://www.who.int/es/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health")

# Load the documents from a CSV file
documents = text_splitter.split_documents(docs)
print("Document splitted")

# Create a vector store
vector = FAISS.from_documents(documents, embeddings)
print("Vector created")

# Save the vector store
vector.save_local("vector_store")
print("Vector saved")

# Load the vector store
retriever = vector.as_retriever()
print("Retriever created")

# Create a prompt
prompt = ChatPromptTemplate.from_template(
"""Respond only in spanish. You are called AutoPaperwork and you are an expert on air quality, first of all, present yourself shortly. Read the context carefully and with attention. Then, do what the input tells you to do based only on the context:

<context>
{context}
</context>

Question: {input}""")

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)
print("Document chain created")

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("Retrieval chain created")

# Receive user input
prompt_text = receive_user_input()
print("Prompt text:")

# Invoke the retrieval chain
response = retrieval_chain.invoke({"input": prompt_text})
print("Response:")
print(response["answer"])
