from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm = Ollama(model="llama3:instruct")

loader = WebBaseLoader("https://es.wikipedia.org/wiki/Constantino_I")
docs = loader.load()

output_parser = StrOutputParser()

embeddings = OllamaEmbeddings()

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
    chunk_size=500,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,)

documents = text_splitter.split_documents(docs)
print(documents)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the following question in spanish based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "cuentame todo lo que sepas de la primera tetrarquia"})
print(response["answer"])
