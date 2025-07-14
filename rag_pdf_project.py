from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
import shutil

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, replace * with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings once
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=""   # replace with your OpenAI API key
)

vector_store = None
retriever = None

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded PDF to disk
    pdf_path = Path(f"temp_{file.filename}")
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load PDF content
    loader = PyPDFLoader(file_path=str(pdf_path))
    docs = loader.load()

    # Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    global vector_store, retriever
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="DSA_Learning",
        embedding=embedder,
    )

    retriever = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="DSA_Learning",
        embedding=embedder,
    )

    return {"message": "PDF uploaded and processed."}

@app.post("/ask")
async def ask(question: str = Form(...)):
    global retriever

    if retriever is None:
        return {"answer": "No PDF has been uploaded yet."}

    relevant_chunks = retriever.similarity_search(query=question, k=4)

    if not relevant_chunks:
        return {"answer": "No relevant information found."}

    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    system_prompt = f"""
        You are a helpful assistant that answers questions based on the provided context and DSA book provided.
        Rules:
        - Always provide a concise and accurate answer based on the context.
        - If the context does not contain enough information, respond in a funny and kind way with a smile emoji.
        - Keep answers ≤ 400 words if possible.
        - Give code examples if question is about code.
        -You must ONLY answer using the provided context below.
        If the context does NOT contain the answer, reply exactly:
        "I am sorry, I don't know the answer to that question based on the uploaded document. 😊"
        Do NOT answer from your own knowledge.

        Context:
        {context}
    """

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=""   # replace with your OpenAI API key
    )

    response = model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])

    return {"answer": response.content}
