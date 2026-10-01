import os

from dotenv import load_dotenv


class LLMParameters:
    def __init__(self,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str,
    ):
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key

    @classmethod
    def from_env(cls) -> LLMParameters:
        load_dotenv()

        return cls(
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
        )
