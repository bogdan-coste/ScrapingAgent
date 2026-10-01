import json
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCall

from ..generation.base_agent import BaseAgent
from .skill_loader import SkillManager

_SKILLS_DIR = Path(__file__).parent / "skills"
SUBAGENT_SKILLS_DIR = str(_SKILLS_DIR / "subagent")


class Agent:
    """
        A subagent.

        Runs the LLM/tool loop for the single task it is given and returns the
        result. It has no orchestrator duties: no objective file, no subagents
        of its own.
    """

    def __init__(
        self,
        agent_llm: BaseAgent,
        shared_resources: dict[str, Any] | None = None,
        skills_path: str | None = None,
        required_skills: list[str] | None = None,
        max_steps: int = 10,
    ):
        self.agent_llm = agent_llm
        self.max_steps = max_steps
        self.skills_path = skills_path or SUBAGENT_SKILLS_DIR

        # Values passed here (e.g. {"url": link}) land in each skill module's
        # namespace when it loads, so this subagent's skills see its own url.
        self.skills = SkillManager(
            self.skills_path,
            required_skills,
            shared_resources=shared_resources or {},
        )
        self.skills.load_skills()

    async def run(self, params: Any = None) -> str:
        """
            Run the loop for the given request and return the result.
        """
        messages: list[Any] = list(self.agent_llm.build_prompt(params))

        return await self._loop(messages)

    def _tools(self) -> list[Any]:
        return list(self.skills.tools_schema)

    async def _loop(self, messages: list[Any]) -> str:
        tools = self._tools()

        for _ in range(self.max_steps):
            message = self.agent_llm.ask_with_tools(messages, tools, tool_choice="auto")
            messages.append(message)

            tool_calls = message.tool_calls or []
            if not tool_calls:
                return message.content or ""

            for tool_call in tool_calls:
                if tool_call.type != "function":
                    continue

                try:
                    result = await self._execute_tool(tool_call)
                except Exception as exc:
                    # a failing tool must not kill the run: report it back so the
                    # LLM can adapt (retry, pick another link, or just finish)
                    result = f"Error: tool '{tool_call.function.name}' failed: {exc}"

                print(f"   [Tool] {tool_call.function.name} → {str(result)[:200]}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        return "Max steps reached."

    async def _execute_tool(self, tool_call: ChatCompletionMessageFunctionToolCall) -> Any:
        """
            Run a single tool call and return its result.
        """
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return "Error: invalid JSON in tool arguments."

        return await self.skills.aexecute_tool(tool_call.function.name, arguments)
