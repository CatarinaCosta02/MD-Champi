from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

PATH = "data/plan/3daybeginners.pdf"

def load_docs():
    # Load pdfs
    doc_loader = PyPDFLoader(PATH)
    pages = doc_loader.load_and_split()
    print("Pages: ", len(pages))
    return pages

def split_docs(pages):
    # Split the pdf data
    chunk = chunk = pages[0].page_content
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(chunk)

    # Char-level splits
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    print(splits)
    return splits
#----------------TO DO-----------------------
def csv_loader(PATH):
    loader = CSVLoader(PATH, encoding='utf-8')
    data = loader.load()

    print(data)

def pinecome_populate():
    pass
def chunks_id():
    pass
def clear_db():
    pass


def main():
    pages = load_docs()
    splits = split_docs(pages)
    csv_loader()

if __name__ == "__main__":
    main()