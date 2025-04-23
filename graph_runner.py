from langgraph.graph import StateGraph, END             # LangGraph core components for defining stateful workflows
from langchain_core.runnables import RunnableLambda     # Allows wrapping custom Python logic as LangChain runnables
from agents.reserach_agent import build_research_agent  # Import the research agent
from agents.writer_agent import build_writer_agent      # Import the writer agent

# Define a simple state structure using Python dict
class State(dict):
    """
    Custom state class extending Python's dictionary.
    Used to maintain state between graph nodes.
    """
    pass  

# Define the 'research' node in the LangGraph
def research_node(state):
    """
    Node that performs web research using the research agent.

    Args:
        state (dict): The current state containing at least a 'query' key.

    Returns:
        dict: Updated state with added 'context' key.
    """
    
    # Ensure the required key exists
    if "query" not in state:
        raise KeyError("The key 'query' is missing in the state.")

    query = state["query"]
    # Invoke the research agent with the query
    context = build_research_agent().invoke(query)
    
    # Debug logging: Print what the state looks like
    print(f"Research Node - state: {state}")
    
    return {"query": query, "context": context}

# Define the 'writer' node in the LangGraph
def writer_node(state):
    """
    Node that generates the final answer using the writer agent.

    Args:
        state (dict): The current state containing 'query' and 'context'.

    Returns:
        dict: Updated state with the added 'answer' key.
    """
    # Ensure both keys exist
    if "query" not in state or "context" not in state:
        raise KeyError("Either 'query' or 'context' is missing in the state.")

    query = state["query"]
    context = state["context"]
    # Invoke the writer agent with query + context
    answer = build_writer_agent().invoke({"query": query, "context": context})
    
    # Debug logging: Print what the state looks like
    print(f"Writer Node - state: {state}")
    
    return {"query": query, "context": context, "answer": answer}

# Build LangGraph
# ----------------- LangGraph Construction -----------------

# Initialize the LangGraph using the custom state structure
graph = StateGraph(State)
graph.add_node("research", RunnableLambda(research_node))
graph.add_node("writer", RunnableLambda(writer_node))

# Define the flow of execution
graph.set_entry_point("research")        # Start at the research node
graph.add_edge("research", "writer")     # After research, go to writer
graph.add_edge("writer", END)            # Writer is the final node

# Compile the graph into an executable app
app = graph.compile()

# ----------------- Public Interface -----------------

# ✅ THIS is the function you're importing in Streamlit
def run_langgraph_pipeline(user_query):
    """
    Entry function for running the full LangGraph pipeline.

    Args:
        user_query (str): The user's research question or URL.

    Returns:
        str: The final generated answer from the pipeline.
    """
    initial_state = {"query": user_query}
    
    # Debug logging: Print the initial state
    print(f"Initial state: {initial_state}")
    
    final_state = app.invoke(initial_state)
    
    # Debug: Print the final state
    print(f"Final state: {final_state}")
    
    # Return the final answer or a fallback message
    return final_state.get("answer", "No answer found")

