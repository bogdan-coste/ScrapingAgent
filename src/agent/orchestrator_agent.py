import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCall

from ..generation.base_agent import BaseAgent
from ..generation.generic_agent import GenericAgent, TaskRequest
from ..schemas.models import LLMParameters
from .skill_loader import SkillManager

_SKILLS_DIR = Path(__file__).parent / "skills"
ORCHESTRATOR_SKILLS_DIR = str(_SKILLS_DIR / "orchestrator")

class OrchestratorAgent:

    def __init__(
        self,
        agent_llm: BaseAgent,
        objective_file: str | None = None,
        required_skills: list[str] | None = None,
        skills_path: str | None = None,
        name: str | None = None,
        max_steps: int = 10,
    ):
        self.agent_llm = agent_llm
        self.objective_file = objective_file
        self.name = name
        self.max_steps = max_steps
        self._created = 0

        if skills_path is None:
            self.skills_path = ORCHESTRATOR_SKILLS_DIR
            
        self.skills_path = skills_path

    async def run(self, params: Any = None) -> str:
        """
            Run the agent loop.

            Coordinator: works towards the objective read from
            ``objective_file`` and ignores ``params``.
            Subagent:    ``params`` is the request handed over by the
            coordinator (the agent's own request type).
        """
        if self.mode is AgentMode.COORDINATOR:
            messages: list[Any] = [
                {"role": "system", "content": self.agent_llm.system_prompt},
                {"role": "user", "content": self._load_objective()},
            ]
        else:
            messages = list(self.agent_llm.build_prompt(params))

        report = await self._loop(messages)

        if self.name:
            self.results.write(self.name, report)

        return report

    def _load_objective(self) -> str:
        if not self.objective_file:
            raise ValueError("Coordinator mode requires an objective_file.")

        return Path(self.objective_file).read_text(encoding="utf-8")

    def _tools(self) -> list[Any]:
        tools = list(self.skills.tools_schema)
        if self.mode is AgentMode.COORDINATOR and self.subagents:
            tools.append(self._read_results_tool_schema())
        return tools

    def _read_results_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_results",
                "description": "Read the findings reported by the subagents spawned so far.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

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

                result = await self._execute_tool(tool_call)
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

        name = tool_call.function.name

        if name == "read_results":
            return self._results_report()

        return await self.skills.aexecute_tool(name, arguments)

    async def _run_subagent(self, name: str, params: dict[str, Any]) -> str:
        """
            Run a registered subagent and return the report it produces.

            Exposed to skill files as the shared ``run_subagent`` helper, so a
            skill can spawn a subagent without importing anything.
        """
        subagent = self.subagents.get(name)
        if subagent is None:
            return f"Error: unknown subagent '{name}'."

        request = subagent.request_factory(**params)

        agent = Agent(
            subagent.agent_llm,
            AgentMode.SUBAGENT,
            name=subagent.name,
            results=self.results,
            skills_path=subagent.skills_path,
            required_skills=subagent.required_skills,
        )

        return await agent.run(request)

    def _results_report(self) -> str:
        """
            Format everything the spawned subagents have reported so far.
        """
        if not self.results:
            return "No subagent has reported yet."

        return "\n\n".join(
            f"### {key}\n{report}" for key, report in self.results.all().items()
        )
