from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from agents.reserach_agent import build_research_agent
from agents.writer_agent import build_writer_agent

# Define state structure
class State(dict):
    pass  # Using simple dict-style state

# Define nodes with better debugging info
def research_node(state):
    print(f"Research Node - Received state: {state}")
    
    # Ensure query exists in the state
    if "query" not in state:
        raise KeyError("The key 'query' is missing in the state.")

    query = state["query"]
    context = build_research_agent().invoke(query)

    # Ensure the result is valid and context is present
    print(f"Research Node - Returning state: {{'query': {query}, 'context': {context}}}")
    
    return {"query": query, "context": context}

def writer_node(state):
    print(f"Writer Node - Received state: {state}")
    
    # Ensure query and context exist in the state
    if "query" not in state or "context" not in state:
        raise KeyError("Either 'query' or 'context' is missing in the state.")

    query = state["query"]
    context = state["context"]
    answer = build_writer_agent().invoke({"query": query, "context": context})

    # Ensure the result is valid and answer is present
    print(f"Writer Node - Returning state: {{'query': {query}, 'context': {context}, 'answer': {answer}}}")
    
    return {"query": query, "context": context, "answer": answer}

# Build LangGraph
graph = StateGraph(State)
graph.add_node("research", RunnableLambda(research_node))
graph.add_node("writer", RunnableLambda(writer_node))

graph.set_entry_point("research")
graph.add_edge("research", "writer")
graph.add_edge("writer", END)

# Compile
app = graph.compile()

# ✅ THIS is the function you're importing in Streamlit
def run_langgraph_pipeline(user_query):
    initial_state = {"query": user_query}

    # Debug: Print the initial state being passed to the pipeline
    print(f"Initial state: {initial_state}")

    final_state = app.invoke(initial_state)

    # Debug: Print the final state after invoking the pipeline
    print(f"Final state: {final_state}")
    
    return final_state.get("answer", "No answer found")



# import streamlit as st
# from graph_runner import run_langgraph_pipeline
# from dotenv import load_dotenv, find_dotenv

# # Load environment variables from .env
# load_dotenv(find_dotenv())

# # Set up Streamlit page configuration
# st.set_page_config(page_title="AI Research Agent", layout="wide")

# # Display title
# st.title("🧠 Deep Research AI Agent")

# # Input for user query or URL
# user_query = st.text_input("Enter your research question or URL", "")

# # When the button is clicked, run the agent
# if st.button("Run Research"):
#     # Check if the user entered something valid (non-empty)
#     if user_query.strip():
#         with st.spinner("Running agents..."):
#             try:
#                 # Call the pipeline function and get the result
#                 final_answer = run_langgraph_pipeline(user_query)

#                 # Display the final answer from the research pipeline
#                 st.markdown("### ✅ Final Answer")
#                 st.write(final_answer)

#             except KeyError as e:
#                 # Handle the KeyError if the query is missing or there's an issue
#                 st.error(f"Error: The key 'query' was not found. Please check your input.")
#             except Exception as e:
#                 # Catch all other exceptions (e.g., API errors)
#                 st.error(f"Something went wrong: {str(e)}")

#     else:
#         # If the input is empty or invalid, show a warning
#         st.warning("Please enter a valid query or URL.")
