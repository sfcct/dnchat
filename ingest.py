# import dataiku
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# FILE_NAME = "housing.txt" 
FILE_NAME = "DNVGL-ST-F101.pdf"
# Load the PDF file and split it into smaller chunks
# docs_folder = dataiku.Folder("docs") # Replace with your input folder id
f_path = r'C:\D\Main\IIMC _ APDS\NLP\RAG_chat_deploy\docs\\'+FILE_NAME
if FILE_NAME.split('.')[1]=='pdf':loader = PyPDFLoader(f_path)
if FILE_NAME.split('.')[1]=='txt':loader = TextLoader(f_path)
doc = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
chunks = text_splitter.split_documents(doc)

# Retrieve embedding function from code env resources
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)

# Index the vector database by embedding then inserting document chunks
# vector_db_folder = dataiku.Folder("xxx") # Replace with your output folder id 
vector_db_path = r'C:\D\Main\IIMC _ APDS\NLP\RAG_chat_deploy\db'
db = Chroma.from_documents(chunks,
                           embedding=embeddings,
                           persist_directory=vector_db_path)

# Save vector database as persistent files in the output folder
db.persist()

query = "What is critical pressure?"
matching_docs = db.similarity_search(query)

print (matching_docs[0])
# print (matching_docs[1])
# print (matching_docs[2])

