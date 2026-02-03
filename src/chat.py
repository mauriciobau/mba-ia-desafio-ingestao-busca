"""
Chat CLI for Semantic Search.

This script provides an interactive command-line interface for
querying the ingested PDF documents using natural language.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from search import search, format_context, PROMPT_TEMPLATE

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-2.0-flash-lite"


def create_llm():
    """
    Initialize the LLM for generating responses.
    
    Returns:
        ChatGoogleGenerativeAI: The initialized LLM instance.
    """
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )


def ask_question(llm, question: str) -> str:
    """
    Process a user question and generate a response.
    
    Args:
        llm: The LLM instance.
        question: The user's question.
    
    Returns:
        The LLM's response based on the retrieved context.
    """
    # 1. Search for relevant documents
    results = search(question, k=10)
    
    # 2. Format context from results
    context = format_context(results)
    
    # 3. Build prompt with context and question
    prompt = PROMPT_TEMPLATE.format(
        contexto=context,
        pergunta=question
    )
    
    # 4. Get response from LLM
    response = llm.invoke(prompt)
    
    return response.content


def main():
    """
    Main CLI loop for user interaction.
    """
    print("=" * 60)
    print("🤖 Chat de Busca Semântica")
    print("=" * 60)
    print("Digite sua pergunta ou 'sair' para encerrar.\n")
    
    # Initialize LLM
    try:
        llm = create_llm()
    except Exception as e:
        print(f"❌ Erro ao inicializar o LLM: {e}")
        print("Verifique sua GOOGLE_API_KEY no arquivo .env")
        return
    
    while True:
        try:
            # Get user input
            question = input("\n📝 Faça sua pergunta: ").strip()
            
            # Check for exit commands
            if question.lower() in ["exit", "quit", "sair", "q"]:
                print("\n👋 Até logo!")
                break
            
            # Skip empty input
            if not question:
                continue
            
            # Process question and display answer
            print("\n🔍 Buscando informações...")
            answer = ask_question(llm, question)
            print(f"\n💬 Resposta: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro ao processar pergunta: {e}")


if __name__ == "__main__":
    main()