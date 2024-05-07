import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
import re


PATH = "data/plan/3daybeginners.pdf"

def load_docs():
    doc_loader = PyPDFLoader(PATH)
    pages = doc_loader.load_and_split()
    print("Pages: ", len(pages))
    print(f'*from load_docs type(pages): {type(pages)}')
    return pages


# paths = ["data/plan/3daybeginners.pdf", 
#          "data/plan/3dayworkoutroutineanddietforbeginners.pdf",
#          "data/plan/8weekbeginnerfatlossworkoutforwomen_0.pdf",
#          "data/plan/8weekbeginnerworkoutforwomen.pdf",
#          "data/plan/12weekfullbodyworkoutroutineforbeginners.pdf",
#          "data/plan/startfromscratch.pdf",
#          "data/plan/thebest15minutewarmups.pdf"]

# def load_docs(paths):
#     all_pages = []
#     for path in paths:
#         doc_loader = PyPDFLoader(path)
#         pages = doc_loader.load_and_split()
#         print(f"Pages from '{path}': {len(pages)}")
#         all_pages.extend(pages)
#     return all_pages



def split_docs(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = [page.page_content for page in documents]
    documents = text_splitter.create_documents(texts)
    # for doc in documents:
    #     print(f'\n{doc.page_content}')
    return documents


def extract_metadata(doc_content): ############################################### CONTINUAR
    metadata = {}
    lines = doc_content.split('\n')

    # Extract day of the week
    metadata['day'] = lines[0].split(' - ')[0]
    metadata['exercise_name'] = lines[0].split(' - ')[1]

    # Extract notes
    notes_index = lines.index('Notes')
    notes = '\n'.join(lines[notes_index + 1:])
    metadata['notes'] = notes

    return metadata


def transform_text(text):    
    # Transformar o texto
    text = text.lower()  # Converter para minúsculas
    text = (re.sub(r'[^\w\s]', '', text)).strip()
    return text


def save_docs_to_txt(documents, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(f'{doc}\n')



def pinecome_populate():
    pass
def chunks_id():
    pass
def clear_db():
    pass


def main():
    pages = load_docs()
    splits = split_docs(pages)
    for doc in splits:
        print(f'\ndoc.page_content: {doc}')
        metadata = extract_metadata(doc.page_content)
        print(metadata)
        print()
    

if __name__ == "__main__":
    main()
