import sys
import os
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader, Settings, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

def query_wowhead(query: str):
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()
        Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
        Settings.llm = Ollama(model="mistral", request_timeout=30.0)
        index = VectorStoreIndex.from_documents(
            documents,
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        ServiceContext.from_defaults(embed_model="local:BAAI/bge-small-en-v1.5", llm=Ollama(model="mistral", request_timeout=30.0))
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

if __name__ == "__main__":
    query = sys.argv[1]
    response = query_wowhead(query)
    print(response)
