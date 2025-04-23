from agents.reserach_agent import build_research_agent
from agents.writer_agent import build_writer_agent

def main():
    print("🔍 Research System")
    print("1. Enter a question/topic for web research")
    print("2. Provide a specific URL to summarize")
    choice = input("Choose [1/2]: ")

    query = input("Enter your query or URL: ")

    print("\n🔎 Collecting info...")
    research_agent = build_research_agent()
    research_output = research_agent.invoke(query)

    print("\n✍️ Composing final answer...")
    writer_agent = build_writer_agent()
    final_answer = writer_agent.invoke({
        "query": query,
        "context": research_output
    })

    print("\n✅ Final Answer:\n")
    print(final_answer)

if __name__ == "__main__":
    main()
