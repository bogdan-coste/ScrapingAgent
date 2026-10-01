import json
import os
import glob
import importlib.util
import inspect
from typing import Dict, Any, cast


class SkillManager:
    def __init__(self, skills_dir, required_skills, shared_resources=None):
        self.skills_dir = skills_dir
        self.required_skills = required_skills
        self.shared_resources = shared_resources or {}
        self.tools_schema = []
        self.function_map = {}
        self.skill_docs = {}
        self.func_to_skill_map = {}

    def load_skills(self):
        """
        This function lazy loads the skills
        from the skills directory.
        """

        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)

        # taking the paths from all the .py files within the working directory
        py_files = glob.glob(os.path.join(self.skills_dir, "*.py"))

        # iterating through
        for py_path in py_files:

            # extracting the name of the Python file from the path, removing the extension
            skill_name = str(os.path.basename(py_path)).replace(".py", "")

            # filtering by required skills
            if self.required_skills is not None and skill_name not in self.required_skills:
                continue

            # building the path to the .md file by merging the path of the base skills directory and the name of the file by adding
            # the .md extension
            md_path = os.path.join(self.skills_dir, f"{skill_name}.md")

            # putting the skill description from the .md file in the dictionary for each skill
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    self.skill_docs[skill_name] = f.read()

            # reading file's metadata first
            spec = importlib.util.spec_from_file_location(skill_name, py_path)

            if spec is None:
                print(f"Warning: could not load spec for {skill_name}, skipping.")
                continue

            # Creates an empty, isolated namespace in memory where the module's code will live.
            module = importlib.util.module_from_spec(spec)

            # Executes the code from the file (py_path) into the empty module,
            # populating it with the functions, variables and classes defined in that file.
            if spec.loader is not None:
                spec.loader.exec_module(module)

            # Put the shared values (e.g. this subagent's url) into the skill's
            # namespace. Done *after* exec so a skill can declare a default
            # (e.g. `url: str = ""`) that these values then override.
            module.__dict__.update(self.shared_resources)

            # Register top‑level functions (your original code) ---
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_") or name == "initialize":
                    continue

                # only register functions *defined* in this skill file, not the
                # helpers it happens to import (e.g. `extract_links`)
                if func.__module__ != module.__name__:
                    continue

                self._register_tool(name, func, skill_name)

            # Register public methods of public classes ---
            for cls_name, cls_obj in inspect.getmembers(module, inspect.isclass):
                # Skip private / special classes (e.g. _Internal)
                if cls_name.startswith("_"):
                    continue

                if cls_obj.__module__ != module.__name__:
                    continue

                # Try to instantiate the class.
                # If the __init__ requires arguments, you'll need to supply them.
                # Solution: make the class accept a default config or use environment variables.
                try:
                    instance = cls_obj()
                except Exception as e:
                    print(f"Warning: could not instantiate class {cls_name} in {skill_name}: {e}")
                    continue

                # Walk through the instance's methods and register the public ones
                for m_name, method in inspect.getmembers(instance, inspect.ismethod):
                    if m_name.startswith("_"):
                        continue
                    # `method` is a bound method; inspect.signature() will automatically drop 'self'
                    self._register_tool(m_name, method, skill_name)

    def _register_tool(self, func_name: str, func: Any, skill_name: str):
        """
        Registers a function as a callable tool for the LLM.

        Builds the JSON schema for the tool from the function's type hints and docstring,
        appends it to tools_schema, and maps the function name to both
        the callable function (function_map) and its parent skill (func_to_skill_map).
        """
        if func_name in self.function_map:
            print(f"Warning: duplicate tool '{func_name}' in '{skill_name}', skipping.")
            return

        # this function return the complete signature of the function
        # the name of the parameters, type hints and the default values
        sig = inspect.signature(func)

        # extracts the docstring of the function
        doc = func.__doc__ or "No documentation."

        # iterating through the list of the parameters
        # obtained from sig
        params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            param_type = "string"
            param_desc = "Parameter"

            # param.annotation is the type hint associated
            # with the parameter;
            # param_type si param_desc, both strings
            # will be read and interpreted by the LLM.
            if param.annotation == list:
                param_type = "array"
                param_desc = 'String array'
            elif param.annotation == dict:
                param_type = "object"
                param_desc = "JSON object"
            elif param.annotation == int:
                param_type = "integer"
            elif param.annotation == bool:
                param_type = "boolean"

            # this is the dictionary that contains the parameters
            # with a description in string format.
            cast(dict[str, Any], params["properties"])[param_name] = {
                "type": param_type,
                "description": param_desc
            }

            if param.default == inspect.Parameter.empty:
                cast(list[str], params["required"]).append(param_name)

        brief = self.skill_docs.get(skill_name, "")
        if brief:
            first_line = brief.strip().splitlines()[0].strip()
            description = first_line
        else:
            description = doc

        self.tools_schema.append({
            "type": "function",
            "function": {
                "name": func_name,
                "description": description,
                "parameters": params,
            }
        })

        # dictionary with format name: real_function
        # used when the LLM calls a tool
        self.function_map[func_name] = func

        # dictionary with format function_name : skill_name
        # used to know in which file is the function located
        self.func_to_skill_map[func_name] = skill_name

    def get_skill_manual(self, func_name: str) -> str:
        """
        Returns the Markdown(MD) file of the function's manual.
        """
        skill_name = self.func_to_skill_map.get(func_name)

        if skill_name and skill_name in self.skill_docs:
            return f"\n--- MANUAL is active: {str(skill_name).upper()} ---\n{self.skill_docs[skill_name]}\n"
        return ""

    def execute_tool(self, name: str, args: Dict) -> Any:
        """
        Function to execute another function from a module.
        """
        if name in self.function_map:
            return self.function_map[name](**args)
        return f"Error: Tool {name} not found."

    async def aexecute_tool(self, name: str, args: dict) -> Any:
        """
        Execute a tool, awaiting the result when the skill function is async.
        """
        if name not in self.function_map:
            return f"Error: Tool {name} not found."

        result = self.function_map[name](**args)

        # skills may be sync or async; await only when needed
        if inspect.isawaitable(result):
            result = await result

        return result
