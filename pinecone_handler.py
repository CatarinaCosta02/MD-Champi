import datetime
import gzip
import re
import sys
import uuid
import numpy as np
from pinecone import ServerlessSpec, Pinecone
import os
from dotenv import load_dotenv
import time
from langchain_community.embeddings import OllamaEmbeddings
from loader import get_pdfs, load_docs, split_docs, extract_metadata
import json
from tqdm import tqdm
from sklearn.decomposition import PCA



def get_index(index_name):

    api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=api_key, pool_threads=30)

    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    index_dimension = 4096
    # check if index already exists (it shouldn't if this is first time)
    # a metrica que este usa é o dotproduct
    if index_name not in existing_indexes:
        # if does not exist, create index
        pc.create_index(
            name=index_name,    # index_name é passado corretamente como uma string
            dimension=index_dimension,     # dimensionality of ada 002
            metric='dotproduct',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # for index in pc.list_indexes():
    #     print(f'index name: {index['name']}')

    # connect to index
    index = pc.Index(index_name)
    # print(f'index: {index}')
    time.sleep(1)

    # # apagar os embedded vectors do index
    # index.delete(delete_all=True)

    return index



def main():

    directory_path = r'.\data\\plan\\'
    pdf_files = get_pdfs(directory_path)

    load_dotenv()

    index_name = 'champi'
    index = get_index(index_name)
    print(f'index \'{index_name}\': {index}')

    # view index stats
    old_stats = index.describe_index_stats()
    print(f'\nindex stats: {old_stats}\n')

    pages = load_docs(pdf_files)
    documents = split_docs(pages)

    index_dimension = 4096

    vectors_dictionary = {}
    vector = []
    vector_count = 0
    line_count = 0
    for doc in documents:
        for line in doc.page_content.split('\n'):

            line_count += 1

            line = (str(line)).lower()
            line = (re.sub(r'[^\w\s]', '', line)).strip()

            vector.append(line)

            # Verifica se o vetor excede o limite de tamanho
            if len(vector) >= index_dimension:
                vector_count += 1
                # Adiciona o vetor completo ao dicionário
                vectors_dictionary[vector_count] = vector
                # Reinicia o vetor para continuar a adição de novas lines
                vector = []

        # Adiciona o vetor final ao dicionário, caso ainda exista
        if vector:
            vectors_dictionary[vector_count] = vector


    # embeddings = OllamaEmbeddings(model="llama3")
    # embedded_vectors = []
    # startTime = datetime.datetime.now().strftime("%H:%M:%S")
    # print(f'Embedding process start time: {startTime}')
    # # Wrap the iterable with tqdm to create a progress bar
    # for _, vector in tqdm(vectors_dictionary.items(), desc="Embedding Progress"):
    #     # Embed the vector
    #     embedded_vector = embeddings.embed_documents(vector)
    #     print(f'embedded_vector: \n{embedded_vector}')
    #     print(f'len(embedded_vector): \n{len(embedded_vector)}')

    #     embedded_vectors.append(embedded_vector)

    # print(f'\nlen(embedded_vectors): {len(embedded_vectors)}')
    # endTime = datetime.datetime.now().strftime("%H:%M:%S")
    # print(f'Embedding process end time: {endTime}')


    # # Write json file
    # file_path = 'embedded_vectors.json'
    # with open(file_path, 'w') as file:
    #     json.dump(embedded_vectors, file, indent=2)

    # Read json file
    with open('embedded_vectors.json', 'r') as file:
        embedded_vectors = json.load(file)


    print(f'len(embedded_vectors) : {len(embedded_vectors)}')
    for vec in embedded_vectors:
        print(f'len(vec) : {len(vec)}\n')


    # # Flatten embedded_vectors if it's a list of lists
    embedded_vectors = [item for sublist in embedded_vectors for item in sublist]

    embedded_ids = [str(uuid.uuid4()) for _ in range(len(embedded_vectors))]

    metadata = []
    for doc in vectors_dictionary.items():
        mtdt = extract_metadata(doc)
        metadata.append(mtdt)


    # print(f'\nlen(embedded_vectors): {len(embedded_vectors)}')
    # for item in embedded_vectors:
    #     print(f'\n\nitem: {item}')
    #     print(f'\nlen(item): {len(item)}')
    #     for subitem in item:
    #         print(f'subitem: {subitem[:2]}... | len(subitem): {len(subitem)}')


    # print(f'\n\nembedded_ids: {embedded_ids}')
    # print(f'\n\nembedded_vectors: {embedded_vectors}')
    # print(f'\n\nmetadata: {metadata}')


    vectors_to_upsert = [{"id": id, "values": vector, "metadata": metadt} for id, vector, metadt in zip(embedded_ids, embedded_vectors, metadata)]
    index.upsert(vectors=vectors_to_upsert)



    print(f'\n------after inserting data------')
    old_stats = index.describe_index_stats()

    # Wait for the vector_count to update
    while True:
        time.sleep(1)  # Wait for a short period before checking again
        current_stats = index.describe_index_stats()
        if old_stats['namespaces']['']['vector_count'] < current_stats['namespaces']['']['vector_count']:  # Check if vector_count has increased
            old_stats = current_stats.copy()  # Update old_stats with the current stats
        else:
            break

    print(current_stats)







if __name__ == "__main__":
    main()
