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


def receive_user_input():
    if len(sys.argv) > 1:
        prompt_text = ' '.join(sys.argv[1:])
    else:
        prompt_text = "Cuentame quien eres brevemente y, al acabar, dime *que puedo hacer por ti*"
    return prompt_text


def load_a_website(website_url):
    loader = WebBaseLoader(website_url)
    docs = loader.load()
    print("Loaded website")
    return docs


def load_a_csv(csv_path):
    loader = CSVLoader(csv_path)
    docs = loader.load()
    print("Loaded CSV")
    return docs


llm = Ollama(model="llama3:instruct")

output_parser = StrOutputParser()

embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
            )

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

docs = load_a_website("https://es.wikipedia.org/wiki/Seann_William_Scott")

documents = text_splitter.split_documents(docs)
print("Document splitted")

vector = FAISS.from_documents(documents, embeddings)
print("Vector created")


retriever = vector.as_retriever()
print("Retriever created")

prompt = ChatPromptTemplate.from_template(
"""Respond only in spanish. You are called AutoPaperwork, first of all, present yourself with shortly. Read the context carefully and with attention. Then, do what the input tells you to do based only on the context:

<context>
{context}
</context>

Question: {input}""")
print("Prompt created")

document_chain = create_stuff_documents_chain(llm, prompt)
print("Document chain created")

retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("Retrieval chain created")

prompt_text = receive_user_input()
print("Prompt text:")

response = retrieval_chain.invoke({"input": prompt_text})
print("Response:")
print(response["answer"])
