from collections.abc import Sequence

from ..llm.llm_client import LLMClient
from ..prompts.prompt_templates import (
    WEB_SCRAPER_SYSTEM_PROMPT,
    build_web_scraper_prompt,
)
from ..schemas.models import LLMParameters


class WebScraperLLM:

    def __init__(self, params: LLMParameters):
        self.llm = LLMClient(params, system_prompt=WEB_SCRAPER_SYSTEM_PROMPT)

    def build_prompt(
        self,
        target_url: str,
        objective: str,
        fields: Sequence[str],
        output_format: str = "JSON",
        additional_instructions: str = "None.",
    ) -> str:

        return build_web_scraper_prompt(
            target_url=target_url,
            objective=objective,
            fields=fields,
            output_format=output_format,
            additional_instructions=additional_instructions,
        )

    def execute(
        self,
        target_url: str,
        objective: str,
        fields: Sequence[str],
        output_format: str = "JSON",
        additional_instructions: str = "None.",
    ) -> str | None:

        return self.llm.ask_llm(
            self.build_prompt(
                target_url=target_url,
                objective=objective,
                fields=fields,
                output_format=output_format,
                additional_instructions=additional_instructions,
            )
        )
