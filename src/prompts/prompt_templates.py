from collections.abc import Sequence
from string import Template


class PromptTemplate:
    """
        A text prompt containing ``$placeholder`` fields.

        Placeholders follow string.Template syntax, so a literal dollar sign
        must be written as ``$$`` and ordinary braces (e.g. in JSON examples)
        need no escaping.
    """

    def __init__(self, template: str):
        self.template = template

    def render(self, **values: str) -> str:
        """
            Fill every placeholder with the matching value.

            Raises KeyError if a placeholder has no corresponding value.
        """
        return Template(self.template).substitute(values)

    def placeholders(self) -> set[str]:
        """
            Return the name of every placeholder in the template.
        """
        return set(Template(self.template).get_identifiers())


WEB_SCRAPER_SYSTEM_PROMPT = """\
                            You are WebScraperAgent, an autonomous agent that extracts structured data
                            from web pages.

                            Your job:
                            1. Obtain the page you were asked for (fetch it yourself if you can, otherwise
                            work with the HTML or text you were given).
                            2. Locate the elements that actually contain the requested data.
                            3. Extract each value and normalise it into clean, typed output.
                            4. Return the result in the requested format.

                            Rules:
                            - Only report data that is present on the page. Never invent or guess values.
                            - If a field cannot be found, set it to null and explain why in the notes.
                            - When you explain how you found a value, prefer stable identifiers (ids,
                            semantic classes, data-* attributes) over brittle positional selectors.
                            - Respect robots.txt, the site's terms of service, and rate limits. Ask for
                            clarification before doing anything that could get the user blocked or that
                            requires credentials you were not given.
                            - When the requested output is JSON, return valid JSON only, with no prose,
                            markdown fences, or trailing commentary.
                            - Always state the URL you scraped and summarise any fields you could not find.
                            """

WEB_SCRAPER_PROMPT_TEMPLATE = PromptTemplate(
        """\
                Scrape the page below and extract the requested data.

                Target URL: $target_url
                Objective: $objective
                Fields to extract:
                $fields

                Output format: $output_format

                Additional instructions:
                $additional_instructions
    """
)


def build_web_scraper_prompt(
    target_url: str,
    objective: str,
    fields: Sequence[str],
    output_format: str = "JSON",
    additional_instructions: str = "None.",
) -> str:
    """
        Render the web scraper prompt template for a single scraping task.
    """
    return WEB_SCRAPER_PROMPT_TEMPLATE.render(
        target_url=target_url,
        objective=objective,
        fields="\n".join(f"- {field}" for field in fields) or "- (not specified)",
        output_format=output_format,
        additional_instructions=additional_instructions,
    )
