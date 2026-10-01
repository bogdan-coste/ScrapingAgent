from dataclasses import dataclass

from openai.types.responses import EasyInputMessageParam, ResponseInputParam

from .base_agent import BaseAgent

WEBPAGE_EXPLORER_SYSTEM_PROMPT = """\
You are WebpageExplorer, an agent that reads a web page and reports what it contains.

Your job:
1. Use the navigation tools to open the page and follow the links you need.
2. Report what the page contains and where the requested information lives.
3. Return your findings in the requested format.

Rules:
- Stay on the website you were given.
- Only report what is actually on the page. Never invent content.
- When you reference something, name it with stable identifiers (ids, semantic
  classes, data-* attributes) or the link href.
- Keep the report concise.
"""


@dataclass
class ExploreRequest:
    page_content: str
    goal: str
    output_format: str = "JSON"
    additional_instructions: str = "None."


class WebpageExplorerLLM(BaseAgent[ExploreRequest]):

    @property
    def system_prompt(self) -> str:
        return WEBPAGE_EXPLORER_SYSTEM_PROMPT

    def build_prompt(self, request: ExploreRequest) -> ResponseInputParam:
        return [
            EasyInputMessageParam(role="system", content=self.system_prompt),
            EasyInputMessageParam(
                role="user",
                content=(
                    f"Goal: {request.goal}\n"
                    f"Output format: {request.output_format}\n"
                    f"Additional instructions: {request.additional_instructions}\n\n"
                    f"Page content:\n{request.page_content}"
                ),
            ),
        ]
