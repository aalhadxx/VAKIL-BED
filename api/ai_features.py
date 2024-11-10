# ai_features.py

import logging
import requests
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.document_loaders import PyPDFLoader  # Import for PDF loading
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import for text splitting
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_together import Together
import os
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logger = logging.getLogger("ChatbotAI")
logger.setLevel(logging.INFO)

TOGETHER_AI_API_KEY = os.getenv('TOGETHER_AI_API_KEY')

class TogetherAIEmbeddings(Embeddings):
    #WhereIsAI/UAE-Large-V1
    def __init__(self, api_key, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        self.api_key = api_key
        self.model = model 
        self.endpoint = "https://api.together.xyz/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts):
        """Embed a list of documents using batch processing."""
        payload = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(
            self.endpoint, headers=self.headers, json=payload
        )
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", [])
            embeddings = [item.get("embedding") for item in data]
            if embeddings and all(embeddings):
                return embeddings
            else:
                raise ValueError("Embeddings not found in response data.")
        else:
            raise Exception(f"Error fetching embeddings: {response.text}")

    def embed_query(self, text):
        """Embed a single query."""
        return self._get_embedding(text)

    def _get_embedding(self, text):
        payload = {
            "model": self.model,
            "input": [text]  # Input as a list
        }
        response = requests.post(
            self.endpoint, headers=self.headers, json=payload
        )
        if response.status_code == 200:
            result = response.json()
            try:
                embedding = result["data"][0]["embedding"]
                return embedding
            except (KeyError, IndexError):
                raise ValueError("Embedding not found in response.")
        else:
            raise Exception(f"Error fetching embedding: {response.text}")

# Initialize the embeddings model
embeddings = TogetherAIEmbeddings(
    api_key=TOGETHER_AI_API_KEY,
    model="WhereIsAI/UAE-Large-V1"
)

# Load your PDF document
pdf_path = "BNS.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

# Build the FAISS index with the new embeddings
try:
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local("F:/coding/chatbots/vector db/new_bns_vector_db")
    db_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    logger.info("FAISS index built and loaded successfully with Together AI embeddings.")
except Exception as e:
    logger.error(f"Error building FAISS index: {e}")
    raise

# Together AI LLM configuration
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=200,
    together_api_key=TOGETHER_AI_API_KEY
)

# Main conversation prompt template
main_prompt_template = """<s>[INST] You are a legal expert chatbot specializing in queries related to the Indian Penal Code.
Respond directly to the question based on the specific section or context provided.
Keep responses short, relevant, and factually accurate.

Context: {context}
Question: {question}

Response: [/INST]"""

# Initialize prompt template
prompt = PromptTemplate(
    template=main_prompt_template,
    input_variables=["context", "question"]
)

# Initialize the QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db_retriever,
    memory=ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        return_messages=True
    ),
    combine_docs_chain_kwargs={"prompt": prompt}
)

def process_legal_query(user_input):
    """Process the user's legal query using the QA chain."""
    logger.info("Processing legal query...")
    try:
        result = qa_chain({"question": user_input})
        response = result.get("answer", "")
        if not response or len(response.strip()) < 10:
            response = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "I encountered an issue while processing your query. Could you please rephrase it?"
