
# Import components for creating an AI agent
from agno.agent import Agent
from agno.models.groq import Groq

# Import components for creating a knowledge base
from agno.knowledge import Knowledge
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.vectordb.lancedb import LanceDb

# Import dotenv to access API keys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# Step 1: Initialize the embedding model to convert text into vectors
embedder = SentenceTransformerEmbedder(
    id="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Create a vector database to store and search embeddings
database = LanceDb(
    uri="data/lancedb",
    table_name="course_knowledge",
    embedder=embedder
)

# Step 3: Connect the vector database to the knowledge base
knowledge = Knowledge(vector_db=database)

# Step 4: Read the PDF, generate embeddings, and store them in LanceDB
knowledge.insert(
    path="data/course_details.pdf"
)

# Step 5: Create an AI agent and connect it to the knowledge base
rag_agent = Agent(
    name="Course Information Agent",
    model=Groq(id="openai/gpt-oss-120b"),
    knowledge=knowledge,
    search_knowledge=True,
    instructions=[
    "Answer questions only using information from the knowledge base.",
    "Always search the knowledge base before answering.",
    "If the requested information is not available in the PDF, respond: 'Sorry, I don't have information about that in the provided documents.'",
    "Do not guess or generate information outside the provided documents."],
    markdown=True,
)

# Step 6: Ask a question about the course PDF
rag_agent.print_response(
    "What is the duration of the Data Science course?"
)
