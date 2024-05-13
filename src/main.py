import sys
import DocumentLoader
import TextProcessor
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


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
docs = DocumentLoader.WebDocumentLoader("https://en.wikipedia.org/wiki/Python_(programming_language)").load()

# Load the documents from a CSV file
documents = text_splitter.split_documents(docs)
print("Document splitted")

# Create a vector store
vector = FAISS.from_documents(documents, embeddings)
print("Vector created")

# Load the vector store
retriever = vector.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)
print("Retriever created")

# Create a prompt
prompt = ChatPromptTemplate.from_template(
"""Respond only in spanish. You are called AutoPaperwork and you are an expert on wikipedia searching, first of all, present yourself shortly. Read the context carefully and with attention. Then, do what the input tells you to do based only on the context:

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

# Invoke the retrieval chain
response = retrieval_chain.invoke({"input": prompt_text})
print("Response:")
print(response["answer"])
