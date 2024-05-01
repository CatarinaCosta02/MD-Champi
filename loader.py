from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PATH = "data/plan/3daybeginners.pdf"

def load_docs():
    doc_loader = PyPDFLoader(PATH)
    pages = doc_loader.load_and_split()
    print("Pages: ", len(pages))
    return pages

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = [page.page_content for page in documents]
    documents = text_splitter.create_documents(texts)

    print(documents[2])
    return documents

def pinecome_populate():
    pass
def chunks_id():
    pass
def clear_db():
    pass


def main():
    pages = load_docs()
    splits = split_docs(pages)

if __name__ == "__main__":
    main()