# Create a new conda environment
conda create -n chatbot python=3.11

# Activate the environment
conda activate chatbot

# Install packages available in conda-forge
conda install -c conda-forge streamlit faiss-cpu pdf2image pytesseract pillow

# Install the remaining packages using pip
pip install "langchain>=0.1.0"  # Core LangChain framework for building LLM applications
pip install "langchain-community>=0.0.10"  # Community integrations for LangChain, like RAG tools
pip install "langchain-google-genai>=0.0.5"  # Google Generative AI (Gemini) integration for LangChain
pip install "google-generativeai>=0.3.0"  # Google's official Python SDK for Gemini models
pip install "langchain-openai>=0.0.2"  # OpenAI integration for LangChain
pip install "langchain-ollama"  # Ollama integration for running local LLMs with LangChain
pip install "openai>=1.3.0"  # OpenAI's official Python SDK for GPT models
pip install "pypdf>=3.15.1"  # Library for reading and extracting text from PDF files
conda activate chatbot
which python # should return /Users/your_username/miniconda3/envs/chatbot/bin/python

# Download and Install Ollama
ollama --version # should return a version number like 0.5.12
ollama pull mistral
ollama run mistral # should return a prompt to enter a message # entry /bye to exit
ollama pull nomic-embed-text

# Test the ollama model
ollama = OllamaLLM(model="mistral")
response = ollama.invoke("What's the capital of France?")
print(response)
python test_with_ollama.py # the expected output is: The capital city of France is Paris.

# Run the Chatbot App
streamlit run chat_with_local_ollama.py
streamlit run chat_with_pdf_ollama.py
streamlit run chat_with_pdf_ollama_with_history.py

# Run Chatbot with Remote LLM
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'

#  Run the Chatbot App
streamlit run chat_with_pdf_gemini.py
streamlit run chat_with_pdf_gemini_with_history.py

# Other Options
# find and replace the OPENAI_API_KEY in chat_with_pdf_openai.py
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'



