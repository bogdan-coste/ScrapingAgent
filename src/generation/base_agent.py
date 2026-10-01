from abc import ABC, abstractmethod
from typing import Any

from openai.types.chat import ChatCompletionMessage
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel

from ..llm.llm_client import LLMClient
from ..schemas.models import LLMParameters


class BaseAgent[ReqT](ABC):

    def __init__(self, params: LLMParameters):
        self.llm = LLMClient(params)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
            The system message sent with every request.
        """

    @abstractmethod
    def build_prompt(self, request: ReqT) -> ResponseInputParam:
        """
            Build the full messages payload (system + user) for a request.
        """

    def ask(self, request: ReqT) -> str | None:
        """
            Run a request and return the raw text response.
        """
        return self.llm.ask_llm(self.build_prompt(request))

    def ask_structured(
        self, request: ReqT, response_format: type[BaseModel]
    ) -> BaseModel | None:
        """
            Run a request and parse the response into a Pydantic model.
        """
        return self.llm.ask_llm_pydantic(self.build_prompt(request), response_format)

    def ask_with_tools(
        self,
        messages: list[Any],
        tools: list[Any],
        tool_choice: str = "auto",
    ) -> ChatCompletionMessage:
        """
            Send a messages payload with tools and return the raw assistant
            message, so callers can read ``tool_calls`` off it.
        """
        return self.llm.chat_with_tools(messages, tools, tool_choice)
