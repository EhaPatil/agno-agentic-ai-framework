from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from agno.team import Team 

load_dotenv()
web_agent = Agent(
    name="Web Agent",
    role="Search the internet for information",
    model=Groq(id="openai/gpt-oss-120b"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources in your answers."],
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Retrieve and analyze financial information",
    model=Groq(id="openai/gpt-oss-120b"),
    tools=[YFinanceTools(
        enable_stock_price=True,
        enable_analyst_recommendations=True,
        enable_company_info=True,
        )
    ],
    instructions=["Use tables to display financial data."],
    markdown=True,
)

agent_team = Team(
    name="Research Team",
    model=Groq(id="openai/gpt-oss-120b"),
    members=[web_agent,finance_agent],
    instructions=[
        "Delegate news searches to the Web Agent.",
        "Delegate financial queries to the Finance Agent.",
        "Always include sources.",
        "Use tables to display financial data.",
    ],
    markdown=True,
    show_members_responses=True
)

agent_team.print_response(
    "Find the latest news about NVIDIA and its current stock price.",
    stream=True,
)