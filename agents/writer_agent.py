# importing important libraries
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Loading environment variables (e.g., GROQ_API_KEY in this case)
load_dotenv()

# Initializing Groq LLM client using the provided API key and model
groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

def build_writer_agent():
    """
    Constructs a writer agent that generates well-structured answers using research context and a query from research_agent.
    andr returns:
        RunnableLambda: A callable LangChain Runnable that accepts a query/context dictionary
                        and returns a written answer with citations.
    """
    
    # Prompt template for formatting the writing task
    prompt = PromptTemplate.from_template(
        """

Taking the question, read the research, and write a clear amd  helpful answer using simple language, 
while ptelling from where my info came from using numbered references.

"""
    )

    def run_writer(input: dict):
        """
        Takes query and context as input, and returns a finalized answer.
        then input (dict): Dictionary with keys "query" and "context".
        Returns:
            str: Generated answer based on the input, formatted with citations.
        """
        formatted = prompt.format(query=input["query"], context=input["context"])
        response = groq_llm.invoke(formatted)
        return response.content

    # Wrap-ups the logic in a LangChain RunnableLambda
    return RunnableLambda(run_writer)
