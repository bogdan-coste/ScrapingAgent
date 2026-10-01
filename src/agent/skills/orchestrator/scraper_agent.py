from collections.abc import Sequence
from dataclasses import dataclass

from openai.types.responses import EasyInputMessageParam, ResponseInputParam

from ..prompts.prompt_templates import (
    WEB_SCRAPER_SYSTEM_PROMPT,
    build_web_scraper_prompt,
)
from .base_agent import BaseAgent


@dataclass
class ScrapeRequest:
    target_url: str
    objective: str
    fields: Sequence[str]
    output_format: str = "JSON"
    additional_instructions: str = "None."


class WebScraperLLM(BaseAgent[ScrapeRequest]):

    @property
    def system_prompt(self) -> str:
        return WEB_SCRAPER_SYSTEM_PROMPT

    def build_prompt(self, request: ScrapeRequest) -> ResponseInputParam:
        return [
            EasyInputMessageParam(role="system", content=self.system_prompt),
            EasyInputMessageParam(
                role="user",
                content=build_web_scraper_prompt(
                    target_url=request.target_url,
                    objective=request.objective,
                    fields=request.fields,
                    output_format=request.output_format,
                    additional_instructions=request.additional_instructions,
                ),
            ),
        ]
