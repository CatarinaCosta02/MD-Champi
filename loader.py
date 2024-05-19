import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
import re
import pandas as pd
import glob



def get_pdfs(directory_path):
    pdf_files = glob.glob(directory_path + '*.pdf')
    pdf_files = [os.path.basename(file) for file in pdf_files]
    pdf_files = ['data/plan/' + pdf for pdf in pdf_files]
    return pdf_files


def load_docs(paths):
    data = []
    for path in paths:
        doc_loader = PyPDFLoader(path)
        info = doc_loader.load_and_split()
        data.extend(info)
    return data


def split_docs(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = [page.page_content for page in documents]
    documents = text_splitter.create_documents(texts)

    return documents



def extract_metadata(doc_content):
    doc_content = str(doc_content).lower()
    doc_content = doc_content.removeprefix("page_content='").strip().rstrip("'")
    metadata = {}

    try:
        # Lista de dias da semana para procurar
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_to_number = {day: i+1 for i, day in enumerate(days_of_week)}
        weeks = ['week 1', 'week 2', 'week 3', 'week 4', 'week 5', 'week 6', 'week 7', 'week 8', 'week 9', 'week 10', 'week 11', 'week 12']

        # Verifica se algum dos dias da semana está presente na linha
        days_found = []
        first_day = True
        for day in days_of_week:
            if day in doc_content:
                days_found.append(day)
        if days_found:
            metadata['days'] = ''
            for day in days_found:
                if first_day:
                    first_day = False
                    metadata['days'] = str(day_to_number[day])
                else:
                    metadata['days'] += ', ' + str(day_to_number[day])
        else:
            metadata['days'] = 'any'

        # Verifica se alguma das semanas está presente na linha
        weeks_found = []
        for week in weeks:
            if week in doc_content:
                weeks_found.append(week)
        if weeks_found:
            for week in weeks_found:
                metadata['week'] = week
        else:
            metadata['week'] = 'any'



                
        df = pd.read_csv('data/datasets/df_exercises.csv')

        exercise_info = {}
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            if row['exercise_name'] != 'none':
                exercise_name = row['exercise_name'].lower()
            if row['type'] != 'none':
                exercise_type = row['type'].lower()
            if row['bodypart'] != 'none':
                body_part = row['bodypart'].lower()
            if row['equipment'] != 'none':
                equipment = row['equipment'].lower()

            exercise_info[exercise_name] = (exercise_type, body_part, equipment)

        found_exercises = []
        found_body_parts = []
        found_equipments = []

        for title, (exercise_type, body_part, equipment) in exercise_info.items():
            # Create a regular expression for each part of the exercise information
            pattern_title = re.compile(r'\b' + title + r'\b')
            pattern_exercise_type = re.compile(r'\b' + exercise_type + r'\b')
            pattern_body_part = re.compile(r'\b' + body_part + r'\b')
            pattern_equipment = re.compile(r'\b' + equipment + r'\b')

            # Check if the line contains the title, exercise type, or body part
            if pattern_title.search(doc_content) or pattern_exercise_type.search(doc_content) or pattern_body_part.search(doc_content):
                # Found an exercise and body part, add them to the lists
                found_exercises.append(exercise_type)
                found_body_parts.append(body_part)
                found_equipments.append(equipment)
            
            if pattern_equipment.search(doc_content):
                found_equipments.append(equipment)

        # Convert lists to sets to remove duplicates
        unique_exercises = set(found_exercises)
        unique_body_parts = set(found_body_parts)
        unique_equipment = set(found_equipments)

        # After the loop, check if any exercises or body parts were found
        if unique_exercises:
            metadata['exercises'] = list(unique_exercises)
        else:
            metadata['exercises'] = 'any'

        if unique_body_parts:
            metadata['body_parts'] = list(unique_body_parts)
        else:
            metadata['body_parts'] = 'any'

        if unique_equipment:
            metadata['equipment'] = list(unique_equipment)
        else:
            metadata['equipment'] = 'any'


        metadata['text'] = f'week: {metadata["week"]}; day: {metadata["days"]}; exercises: {metadata["exercises"]}; body part: {metadata["body_parts"]}; equipment: {metadata["equipment"]}'.replace('[', '').replace(']', '').replace('/', '').replace('\\', '')


    except Exception as e:
        print(f"Erro: {e}")

    return metadata



def transform_text(text):
    # Transformar o texto
    text_array = []
    for t in text:
        t = (str(t)).lower()  # Converter para minúsculas
        t = (re.sub(r'[^\w\s]', '', t)).strip()
        text_array.append(t)
    return text_array


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

    directory_path = r'.\data\\plan\\'
    pdf_files = get_pdfs(directory_path)
    print(f'\nPDF files\n')
    for pdf in pdf_files:
        print(pdf)

    print('\n-------------------------------')
    all_data = load_docs(pdf_files)
    print('Metadata extracted from PDF files')
    splits = split_docs(all_data)
    for doc in splits:
        print(f'\ndoc: \n{doc}')
        metadata = extract_metadata(doc.page_content)
        print(f'metadata: \n{metadata}')


if __name__ == "__main__":
    main()
