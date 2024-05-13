import sys
from document_loaders import DocumentLoader
from text_processors import TextProcessor
from vector_stores import VectorStore
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
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

# Choose the model you want to use for embeddings
embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
            )

# Choose the text splitter you want to use
text_splitter = TextProcessor.RCTS(
    separators=[".", "!", "?", "\n"],
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

# Load the documents from a website (in this case)
crude_docs = DocumentLoader.WebDocumentLoader("https://es.wikipedia.org/wiki/Felipe_I_de_Tarento").load()

# Split the documents into chunks
splitted_documents = text_splitter.process(crude_docs)
print("Document splitted")

# Create a vector store
vector = VectorStore.FAISSVectorStore(embeddings).initialize(splitted_documents)
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
