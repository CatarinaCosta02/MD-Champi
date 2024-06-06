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
    
    load_dotenv()

    api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=api_key, pool_threads=30)

    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    index_dimension = 4096
    # check if index already exists (it shouldn't if this is first time)
    # the metric used is dotproduct
    if index_name not in existing_indexes:
        # if does not exist, create index
        pc.create_index(
            name=index_name,
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


    # connect to index
    index = pc.Index(index_name)
    # print(f'index: {index}')
    time.sleep(1)

    return index



def main():

    directory_path = r'.\data\\plan\\'
    pdf_files = get_pdfs(directory_path)

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

            # checks if vector exceeds size limit
            if len(vector) >= index_dimension:
                vector_count += 1
                # adds a complete vector to the dictionary
                vectors_dictionary[vector_count] = vector
                # resets the vector to continue adding new lines
                vector = []
        # Adds the last vector to the dictionary, if it still exists
        if vector:
            vectors_dictionary[vector_count] = vector


    embeddings = OllamaEmbeddings(model="llama3")
    embedded_vectors = []
    startTime = datetime.datetime.now().strftime("%H:%M:%S")
    print(f'Embedding process start time: {startTime}')
    # Wrap the iterable with tqdm to create a progress bar
    for _, vector in tqdm(vectors_dictionary.items(), desc="Embedding Progress"):
        # Embed the vector
        embedded_vector = embeddings.embed_documents(vector)
        print(f'embedded_vector: \n{embedded_vector}')
        print(f'len(embedded_vector): \n{len(embedded_vector)}')

        embedded_vectors.append(embedded_vector)

    print(f'\nlen(embedded_vectors): {len(embedded_vectors)}')
    endTime = datetime.datetime.now().strftime("%H:%M:%S")
    print(f'Embedding process end time: {endTime}')


    print(f'len(embedded_vectors) : {len(embedded_vectors)}')
    for vec in embedded_vectors:
        print(f'len(vec) : {len(vec)}\n')


    #  Flatten embedded_vectors if it's a list of lists
    embedded_vectors = [item for sublist in embedded_vectors for item in sublist]

    embedded_ids = [str(i + 1) for i in range(len(embedded_vectors))]

    metadata = []
    for doc in vectors_dictionary.items():
        mtdt = extract_metadata(doc)
        metadata.append(mtdt)

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
