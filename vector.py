# %%
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"

from langchain_openai import AzureOpenAIEmbeddings
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(prompt="Langsmith APIKEY:")

    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")



# %%
from langchain_community.document_loaders import PyPDFLoader

file_path = "cv.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)

# %%
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://genai-nexus.int.api.corpinter.net/apikey/",
  #  azure_deployment="gpt-4o-mini",
    api_version="2024-06-01",
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-large"
)

# %%
from pymilvus import Collection, MilvusException, connections, db, utility

conn = connections.connect(host="127.0.0.1", port=19530)

# Check if the database exists
db_name = "mbm_milvus_db"
try:
    existing_databases = db.list_database()
    if db_name in existing_databases:
        print(f"Database '{db_name}' already exists.")

        # Use the database context
        db.using_database(db_name)

        # Drop all collections in the database
        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Collection '{collection_name}' has been dropped.")

        db.drop_database(db_name)
        print(f"Database '{db_name}' has been deleted.")
    else:
        print(f"Database '{db_name}' does not exist.")
        database = db.create_database(db_name)
        print(f"Database '{db_name}' created successfully.")
except MilvusException as e:
    print(f"An error occurred: {e}")


