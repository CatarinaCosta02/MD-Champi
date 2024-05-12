import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
import re
import pandas as pd


# PATH = "data/plan/3daybeginners.pdf"

# def load_docs():
#     doc_loader = PyPDFLoader(PATH)
#     pages = doc_loader.load_and_split()
#     # print("Pages: ", len(pages))
#     # print(f'*from load_docs type(pages): {type(pages)}')
#     return pages


# paths = ["data/plan/3daybeginners.pdf",
#          "data/plan/3dayworkoutroutineanddietforbeginners.pdf",
#          "data/plan/8weekbeginnerfatlossworkoutforwomen_0.pdf",
#          "data/plan/8weekbeginnerworkoutforwomen.pdf",
#          "data/plan/12weekfullbodyworkoutroutineforbeginners.pdf",
#          "data/plan/startfromscratch.pdf",
#          "data/plan/thebest15minutewarmups.pdf"]


def load_docs(paths):
    all_pages = []
    for path in paths:
        doc_loader = PyPDFLoader(path)
        pages = doc_loader.load_and_split()
        print(f"Pages from '{path}': {len(pages)}")
        all_pages.extend(pages)
    return all_pages



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



def extract_metadata(doc_content):
    doc_content = str(doc_content).lower()
    doc_content = doc_content.removeprefix("page_content='").strip().rstrip("'")
    metadata = {}

    print(f'\n\ndoc_content: \n{doc_content}')

    df = pd.read_csv('data/datasets/df_exer.csv')

    # Create an empty dictionary to store the title, type, and body part information
    exercise_info = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        title = row['Title'].lower()
        exercise_type = row['Type'].lower()
        body_part = row['BodyPart'].lower()

        # Add the information to the dictionary
        exercise_info[title] = (exercise_type, body_part)

    try:
        # Lista de dias da semana para procurar
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_to_number = {day: i+1 for i, day in enumerate(days_of_week)}
        weeks = ['week 1', 'week 2', 'week 3', 'week 4', 'week 5', 'week 6', 'week 7', 'week 8', 'week 9', 'week 10', 'week 11', 'week 12']


        # metadata_text = doc_content.split('\n')
        # print(f'\n\n\nmetadata["text"] - {metadata_text[0]}\n\n\n')

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


        # if ' - ' in line:
        #     # Extrai a parte do corpo usando o split e remove espaços extras com strip
        #     metadata['body_part'] = line.split(' - ')[1].strip()
        #     metadata['exercise_type'] = 'any'
        #     break
        # else:
            # print(line)
            # Tenta extrair o nome do exercício, tipo e parte do corpo usando expressões regulares


        found_exercises = []
        found_body_parts = []

        for title, (exercise_type, body_part) in exercise_info.items():
            # Create a regular expression for each part of the exercise information
            pattern_title = re.compile(r'\b' + title + r'\b')
            pattern_exercise_type = re.compile(r'\b' + exercise_type + r'\b')
            pattern_body_part = re.compile(r'\b' + body_part + r'\b')

            # Check if the line contains the title, exercise type, or body part
            if pattern_title.search(doc_content) or pattern_exercise_type.search(doc_content) or pattern_body_part.search(doc_content):
                # Found an exercise and body part, add them to the lists
                found_exercises.append(title)
                found_body_parts.append(body_part)

        # Convert lists to sets to remove duplicates
        unique_exercises = set(found_exercises)
        unique_body_parts = set(found_body_parts)

        # After the loop, check if any exercises or body parts were found
        if unique_exercises:
            metadata['exercises'] = list(unique_exercises)  # Convert back to list if needed
        else:
            metadata['exercises'] = 'any'

        if unique_body_parts:
            metadata['body_parts'] = list(unique_body_parts)  # Convert back to list if needed
        else:
            metadata['body_parts'] = 'any'


        metadata['text'] = f'week: {metadata["week"]}; day: {metadata["days"]}; exercises: {metadata["exercises"]}; body part: {metadata["body_parts"]}'.replace('[', '').replace(']', '').replace('/', '').replace('\\', '')

    except Exception as e:
        print(f"Erro: {e}")

    return metadata



# def extract_metadata(doc_content):
#     doc_content = str(doc_content).lower().removeprefix("page_content='").strip().rstrip("'")
#     metadata = {}

#     print(f'\n\ndoc_content: \n{doc_content}')

#     df = pd.read_csv('data/datasets/df_exer.csv')

#     # Create an empty dictionary to store the title, type, and body part information
#     exercise_info = {}

#     # Iterate over each row in the DataFrame
#     for index, row in df.iterrows():
#         title = row['Title'].lower()
#         exercise_type = row['Type'].lower()
#         body_part = row['BodyPart'].lower()

#         # Add the information to the dictionary
#         exercise_info[title] = (exercise_type, body_part)

#     try:
#         # Extract workout days
#         days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
#         day_to_number = {day: i+1 for i, day in enumerate(days_of_week)}
#         days_found = []
#         for day in days_of_week:
#             if day in doc_content:
#                 days_found.append(day)
#         if days_found:
#             for day in days_found:
#                 metadata['day'] = 'day(s) ' + str(day_to_number[day])
#         else:
#             metadata['day'] = 'any'

#         # Extract target muscle groups and exercises
#         exercises_found = []
#         for title, (exercise_type, body_part) in exercise_info.items():
#             pattern = re.compile(r'\b' + title + r'\b')
#             if pattern.search(doc_content):
#                 exercises_found.append((title, exercise_type, body_part))
#         if exercises_found:
#             metadata['exercises'] = exercises_found
#         else:
#             metadata['exercises'] = 'any'

#         # Extract repetitions and sets
#         pattern_sets_reps = re.compile(r'\d+\s+\d+\s+\d+')
#         if pattern_sets_reps.search(doc_content):
#             metadata['sets_reps'] = pattern_sets_reps.findall(doc_content)
#         else:
#             metadata['sets_reps'] = 'any'

#         # Extract warmup recommendations
#         pattern_warmup = re.compile(r'have a\s+\d+\s+min\s+warmup')
#         if pattern_warmup.search(doc_content):
#             metadata['warmup'] = pattern_warmup.findall(doc_content)[0]
#         else:
#             metadata['warmup'] = 'any'

#         # Extract equipment used
#         pattern_equipment = re.compile(r'(barbell|bodyweight|cables|dumbbells|machines)')
#         if pattern_equipment.search(doc_content):
#             metadata['equipment'] = pattern_equipment.findall(doc_content)
#         else:
#             metadata['equipment'] = 'any'

#         # Extract notes on form and technique
#         pattern_notes = re.compile(r'notes\s+(.*)')
#         if pattern_notes.search(doc_content):
#             metadata['notes'] = pattern_notes.findall(doc_content)[0]
#         else:
#             metadata['notes'] = 'any'

#         # Extract program details
#         pattern_program_details = re.compile(r'(main goal|training level|program duration|days per week|time per workout)')
#         if pattern_program_details.search(doc_content):
#             metadata['program_details'] = pattern_program_details.findall(doc_content)
#         else:
#             metadata['program_details'] = 'any'

#     except Exception as e:
#         print(f"Error: {e}")

#     return metadata




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
    pages = load_docs(paths)
    splits = split_docs(pages)
    for doc in splits:
        # print(f'\ndoc.page_content: {doc}')
        metadata = extract_metadata(doc.page_content)
        print(metadata)


if __name__ == "__main__":
    main()
