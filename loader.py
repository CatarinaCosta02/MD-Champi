import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
import re
import pandas as pd


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


import pandas as pd

def extract_metadata(doc_content):
    doc_content = str(doc_content).lower()
    metadata = {}
    dic1 = {}   
    try:
        # Lista de dias da semana para procurar
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        weeks = ['week 1', 'week 2', 'week 3', 'week 4', 'week 5', 'week 6', 'week 7', 'week 8', 'week 9', 'week 10', 'week 11', 'week 12']
        
        # Itera através de cada linha no conteúdo do documento
        for line in doc_content.split('\n'):
            # Verifica se algum dos dias da semana está presente na linha
            for day in days_of_week:
                if day in line:
                    metadata['day'] = day
                    break  # Pare de procurar após encontrar o dia
            else:
                metadata['day'] = 'any'  # Se nenhum dia for encontrado, atribui 'any'
            
            # Verifica se alguma das semanas está presente na linha
            for week in weeks:
                if week in line:
                    metadata['week'] = week
                    break
            else:
                metadata['week'] = 'any'  # Se nenhuma semana for encontrada, atribui 'any'
            
            # Tenta extrair o nome do exercício
            if ' - ' in line:
                metadata['exercise_name'] = line.split(' - ')[1].strip()
                break  # Sai do loop após encontrar o nome do exercício
            
            # Se 'day' e 'week' forem encontrados, sai do loop
            if 'day' in metadata and 'week' in metadata:
                break

    except Exception as e:
        print(f"Erro: {e}")


    df = pd.read_csv('data/datasets/df_exer.csv')


    unique_types = df['Type'].unique()
    unique_title = df['Title'].unique()
    unique_bodyparts = df['BodyPart'].unique()

    print(unique_types)
    print(unique_title)
    print(unique_bodyparts)

    
    return metadata




# def transform_text(text):    
#     # Transformar o texto
#     text_array = []
#     for t in text:
#         t = (str(t)).lower()  # Converter para minúsculas
#         t = (re.sub(r'[^\w\s]', '', t)).strip()
#         text_array.append(t)
#     return text_array


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
