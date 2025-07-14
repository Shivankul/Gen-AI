from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from collections import defaultdict

# Initialize components
llm = ChatOpenAI()
qdrant_client = QdrantClient(url="http://localhost:6333")
embedding_model = OpenAIEmbeddings()
from qdrant_client.models import VectorParams, Distance

# Check and create collection if it doesn't exist
if not qdrant_client.collection_exists("my_collection"):
    qdrant_client.recreate_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

vector_store = Qdrant(client=qdrant_client, collection_name="my_collection", embeddings=embedding_model)

# === PROMPTS===

# Prompt for LESS Abstraction (zoom in)
less_abstraction_prompt = PromptTemplate(
    input_variables=["query"],
    template="Break the following query into 3 simpler sub-questions to increase clarity: {query}"
)

# Prompt for MORE Abstraction (zoom out)
more_abstraction_prompt = PromptTemplate(
    input_variables=["query"],
    template="Reframe the following specific query into a broader context-based question or topic: {query}"
)

# === CHAINS ===
less_chain = LLMChain(llm=llm, prompt=less_abstraction_prompt)
more_chain = LLMChain(llm=llm, prompt=more_abstraction_prompt)

# === INPUT ===
user_query = input("🧠Write your queries here:")

# === Decompose Queries ===
less_sub_queries = less_chain.run(user_query).split('\n')
more_sub_query = more_chain.run(user_query)  # One broader query

print("🔹 LESS Abstraction Queries (Zoomed In):")
for q in less_sub_queries:
    print(f"- {q}")

print("\n🔸 MORE Abstraction Query (Zoomed Out):")
print(f"- {more_sub_query}")

# === RETRIEVE CHUNKS ===
all_queries = less_sub_queries + [more_sub_query] 
#  less_sub_queries(ye ek list hai like subquer1,subquery2) + [more_sub_query](more subqiry ek single string hai isi wjah se isko list me convert kiya gya hai jisse ye bina error ke add ho jaaye)
retrieved_chunks = []

for query in all_queries:
    docs = vector_store.similarity_search(query, k=5)
    retrieved_chunks.extend(docs)



# === DEDUPLICATE ===
unique_chunks = list({doc.page_content: doc for doc in retrieved_chunks}.values())

# === DISPLAY RESULTS ===
print("\n📄 Retrieved Chunks from Combined Decomposition:")
for doc in unique_chunks:
    print(doc.page_content)
    # ye last loop nahi chal rha hai because mera unique_chunks empty rh rha hai so uske liye usme add krna padega data by method of embedding,qdrant jo previos video me dekha hai 