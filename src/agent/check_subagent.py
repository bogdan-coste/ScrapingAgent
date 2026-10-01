"""
    Run a single subagent directly, outside the orchestrator, to check how it
    performs.

    Run from the project root:

        python -m src.agent.check_subagent
"""

import asyncio

from src.agent.agent import Agent
from src.generation.webpage_explorer_agent import ExploreRequest, WebpageExplorerLLM
from src.schemas.models import LLMParameters


async def main() -> None:
    agent = Agent(
        WebpageExplorerLLM(LLMParameters.from_env()),
        shared_resources={"url": "https://www.google.com/"},
    )
    result = await agent.run(ExploreRequest(page_content="", goal="search for babes-bolyai university, find me the syllabus of the computer science in romanian, undergrad and display all the disciplines from the entire undergrad period"))
    print("RESULT:", result)


asyncio.run(main())
