# Deep-Research-AI-Agent
A dual-agent AI system that performs deep web research using LangChain, LangGraph, and Tavily API. Designed for structured research workflows — from crawling sources to generating final written answers.

🚀 Features
🌐 Web Research Agent: Uses Tavily to collect online data.

✍️ Writer Agent: Summarizes and generates a final answer from research.

🔁 Agent Coordination: Managed using LangGraph for controlled step-by-step execution.

🧱 LangChain Framework: Agents built modularly using LangChain tools.

🖥️ Streamlit UI (optional): Easy-to-use interface to run and view results.

🛠️ Setup Instructions
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/your-username/deep-research-agent.git
cd deep-research-agent
2. Create Virtual Environment & Activate
bash
Copy
Edit
python -m venv agent
agent\Scripts\activate  # Windows

💻 Usage
▶️ Option 1: Run with Streamlit
bash
Copy
Edit
streamlit run app.py
Then open the browser to interactively ask questions and view responses.

▶️ Option 2: Run CLI Tool
bash
Copy
Edit
python main.py
You’ll be prompted to enter a topic or a URL. The agents will run and generate a summary.

📦 Project Structure
graphql
Copy
Edit
deep-research-agent/
│
├── agents/
│   ├── research_agent.py      # Builds the web research agent
│   ├── writer_agent.py        # Builds the answer-writing agent
│
├── app.py                     # Streamlit UI
├── main.py                    # CLI interface
├── graph_runner.py            # LangGraph setup and pipeline
├── .env                       # API keys and secrets
└── requirements.txt
🧪 Technologies Used
🛠️ LangChain

🔄 LangGraph

🌐 Tavily API

📜 Streamlit

☁️ dotenv

📌 Notes
Designed to modularly support more agents in the future.

State transitions and outputs are handled cleanly via LangGraph.

Streamlit interface is optional but enhances usability for demos or non-dev users.
