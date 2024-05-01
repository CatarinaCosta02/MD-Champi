from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings

PATH = "data/plan/3daybeginners.pdf"

# Load the pdfs
doc_loader = PyPDFLoader(PATH)
pages = doc_loader.load_and_split()
print("Pages: ", len(pages))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# Extrai o texto de cada página e cria uma lista de strings
texts = [page.page_content for page in pages]

# Agora, passa a lista de strings para o método create_documents
documents = text_splitter.create_documents(texts)

#DB

from langchain_pinecone import PineconeVectorStore

embeddings = OllamaEmbeddings(model="llama3")

index_name = "md"

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

res = embeddings.embed_documents(texts)
print(len(res), len(res[0]))


#docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

#query = "Give me a three day workout plan for beginners."
#docs = docsearch.similarity_search(query)
#print(docs[0].page_content)
