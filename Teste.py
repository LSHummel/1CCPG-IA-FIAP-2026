from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Carrega o arquivo .env
load_dotenv()

# Pega a chave do .env
api_key = os.getenv("OLLAMA_API_KEY")

# Verifica se a chave foi encontrada
if not api_key:
    raise ValueError("OLLAMA_API_KEY não encontrada no arquivo .env")

# Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é {persona}. Responda sobre {especialidade}."),
    ("human", "{pergunta}"),
])

# Ollama Cloud
llm = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {api_key}"
        }
    }
)

# Parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

# Executa
resposta = chain.invoke({
    "persona": "um chef de culinária brasileira",
    "especialidade": "culinária brasileira",
    "pergunta": "Como faço um bolo de cenoura?",
})

print(resposta)