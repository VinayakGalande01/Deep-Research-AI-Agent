# importing necessary librariess
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableLambda
import os

def build_research_agent():
    """
    Constructs a reserach agent using Tavily for web search and WebBaseLoader for URL summarization.

    Returns:
        RunnableLambda: A callable LangChain Runnable that accepts a query and returns summarized research content.
    """
    
    # Initializing Tavily search tool using environment variable
    tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

    def run_research(query: str):
        """
        Executes research logic based on the type of query.
        further query (str): The user's input, which may be a URL or a search term.
        and Returns:
            str: Summarized content from either a specific URL or multiple Tavily search results.
        """
        
         # Case 1: If query is a full URL, summarize that specific page
        if query.startswith("http://") or query.startswith("https://"):
            # Treating like a single web page to summarize
            loader = WebBaseLoader(query)
            docs = loader.load()
            content = docs[0].page_content[:4000]  # trim to fit
            return f"[1] Summary of {query}:\n\n{content}"

        # Case 2: If query is a keyword/topic, uses Tavily to perform a search
        else:
            # Does web search via Tavily
            results = tavily.run(query)
            summaries = []
            for i, item in enumerate(results):
                content = item.get("content", "")[:1000]
                summaries.append(f"[{i+1}] {content.strip()} (Source: {item['url']})")
            return "\n\n".join(summaries)

    # Returns the logic from LangChain Runnable
    return RunnableLambda(run_research)
