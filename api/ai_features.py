# ai_features.py

import logging
import requests
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    def __init__(self, api_key, model="WhereIsAI/UAE-Large-V1"):
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
            "input": [text]
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

class VectorStoreManager:
    def __init__(self, embeddings_model, vector_store_path, pdf_path):
        self.embeddings = embeddings_model
        self.vector_store_path = vector_store_path
        self.pdf_path = pdf_path
        self.vector_store = None
        self.retriever = None

    def load_or_create_vector_store(self):
        """Load existing vector store or create a new one."""
        try:
            # Try to load existing vector store
            logger.info("Attempting to load existing vector store...")
            self.vector_store = FAISS.load_local(
                folder_path=self.vector_store_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded existing vector store")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}")
            logger.info("Creating new vector store...")
            self.create_new_vector_store()

    def create_new_vector_store(self):
        """Create a new vector store from PDF documents."""
        try:
            # Load and process documents
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_documents = text_splitter.split_documents(documents)

            # Create and save new vector store
            self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
            self.vector_store.save_local(self.vector_store_path)
            logger.info("Successfully created and saved new vector store")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def get_retriever(self):
        """Get or create retriever."""
        if self.vector_store is None:
            self.load_or_create_vector_store()
        
        if self.retriever is None:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
        
        return self.retriever

# Initialize components
embeddings = TogetherAIEmbeddings(
    api_key=TOGETHER_AI_API_KEY,
    model="WhereIsAI/UAE-Large-V1"
)

# Initialize vector store manager
vector_store_manager = VectorStoreManager(
    embeddings_model=embeddings,
    vector_store_path="F:/coding/chatbots/vector db/new_bns_vector_db",
    pdf_path="BNS.pdf"
)

# Configure LLM
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=200,
    together_api_key=TOGETHER_AI_API_KEY
)

# Initialize prompt template
PROMPT = PromptTemplate(
    template="""<s>[INST] You are a legal expert chatbot specializing in queries related to the Indian Penal Code.
Respond directly to the question based on the specific section or context provided.
Keep responses short, relevant, and factually accurate.

Context: {context}
Question: {question}

Response: [/INST]""",
    input_variables=["context", "question"]
)

def get_qa_chain(memory=None):
    """Get a QA chain with optional memory."""
    if memory is None:
        memory = ConversationBufferWindowMemory(
            k=2,
            memory_key="chat_history",
            return_messages=True
        )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store_manager.get_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

def process_legal_query(user_input, memory=None):
    """Process the user's legal query using the QA chain."""
    logger.info("Processing legal query...")
    try:
        # Get a QA chain with the provided memory
        current_qa_chain = get_qa_chain(memory)
        result = current_qa_chain({"question": user_input})
        response = result.get("answer", "")
        if not response or len(response.strip()) < 10:
            response = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "I encountered an issue while processing your query. Could you please rephrase it?"
