import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Get the Pinecone API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Get the Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")