from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()
agent = Agent(
    model=Groq(id="openai/gpt-oss-120b"),
    description="You are a helpful AI assistant.",
    tools=[DuckDuckGoTools()],
    instructions=["Search the web when current information is needed"],
    debug_mode=True,
    markdown=True,
)

agent.print_response("What are the latest developments in AI this week?")