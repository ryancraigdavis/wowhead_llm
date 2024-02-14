import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

def query_wowhead(query: str):
    documents = SimpleDirectoryReader("data").load_data()
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    Settings.llm = Ollama(model="mistral", request_timeout=30.0)
    index = VectorStoreIndex.from_documents(
        documents,
    )
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

if __name__ == "__main__":
    query = sys.argv[1]
    response = query_wowhead(query)
    print(response)
