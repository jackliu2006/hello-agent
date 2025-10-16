# %%
import os
from dotenv import load_dotenv, find_dotenv
from getpass import getpass

# 1. Try to find and load .env file
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
    print(f"Loaded environment variables from: {env_path}")
else:
    print("⚠️  No .env file found — falling back to manual input.")

# 2. Helper function to fetch or prompt
def get_env_var(key, prompt_text=None, secret=False):
    """Return environment variable or prompt user if missing."""
    value = os.getenv(key)
    if not value:
        if secret:
            value = getpass(prompt_text or f"Enter {key}: ")
        else:
            value = input(prompt_text or f"Enter {key}: ")
        os.environ[key] = value  # optionally keep it in memory
    return value

# 3. Use it
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", "OPENAI_API_KEY: ", secret=True)
LANGSMITH_API_KEY = get_env_var("LANGSMITH_API_KEY", "LANGSMITH_API_KEY: ", secret=True)
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")

# %%
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://genai-nexus.int.api.corpinter.net/apikey/",
    api_version="2024-06-01",
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

# %%
from langchain_milvus import Milvus

URI = "./milvus_mcap.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

# %%

import os
from pathlib import Path
from uuid import uuid4
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Folder to read from
base_path = Path("/Users/jackliu2006/workspace/mcap")

# Recursively find all files
all_files = list(base_path.rglob("*"))

docs = []

for file_path in all_files:
    if file_path.is_file():
        try:
            print(f"📄 Loading {file_path.name}...")
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)
            uuids = [str(uuid4()) for _ in range(len(split_docs))]
            vector_store.add_documents(documents=split_docs, ids=uuids)
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")

print(f"✅ Loaded {len(docs)} text documents.")




