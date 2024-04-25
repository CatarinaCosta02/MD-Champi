from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Load the PDF file
loader = PyPDFLoader("data/plan/3daybeginners.pdf")
pages = loader.load_and_split()
pages[0]
#print(pages[0])

chunk = chunk = pages[0].page_content


# Split the pdf data
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
splits

#print(splits)
print(splits[0])
# load dos CSVs

#loader = CSVLoader(file_path='data/datasets/df_exer.csv', encoding='utf-8')
#data = loader.load()
#print('                ')
#print('CVSSSSS')
#print(data)

# função para colocar os dados na base de dados