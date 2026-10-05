from dataclasses import dataclass, field
import importlib
from inspect import Parameter, Signature, iscoroutinefunction
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, Optional, Sequence

import yaml

from pydantic import Field
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from tools.fake_image_detector import gemini_settings

from . import prompts, tools
from .context import SessionContext

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """Represents an agent definition loaded from YAML."""

    name: str
    system_prompt: str
    description: str = ""
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    provider: dict[str, Any] = field(default_factory=dict)


def _load_agent_definition_from_file(agent_file: Path) -> AgentDefinition:
    with open(agent_file, "r") as handle:
        agent_data = yaml.safe_load(handle) or {}

    name = agent_data.get("name")
    system_prompt = agent_data.get("system_prompt")

    if not name:
        raise ValueError(f"Agent file '{agent_file}' is missing required 'name' field")
    if not system_prompt:
        raise ValueError(f"Agent file '{agent_file}' is missing required 'system_prompt' field")

    provider = agent_data.get("provider")
    if not provider:
        raise ValueError(f"Agent file '{agent_file}' is missing required 'provider' field")
    if not provider.get("type"):
        raise ValueError(f"Agent file '{agent_file}' is missing required 'provider.type' field")
    if not provider.get("model"):
        raise ValueError(f"Agent file '{agent_file}' is missing required 'provider.model' field")

    return AgentDefinition(
        name=name,
        system_prompt=system_prompt,
        description=agent_data.get("description") or (system_prompt.splitlines()[0] if system_prompt else ""),
        skills=agent_data.get("skills", []) or [],
        tools=agent_data.get("tools", []) or [],
        provider=provider,
    )


class Agent:
    """An immutable AI agent configured from an agent YAML file."""

    def __init__(
        self,
        definition: AgentDefinition,
    ):
        """
        Initialize an Agent from a parsed definition.
        """
        start = perf_counter()
        self.definition = definition
        self.name = definition.name
        self.base_prompt = definition.system_prompt
        self.skill_names = list(definition.skills)
        self.tool_names = list(definition.tools)
        model, self.model_settings = self._build_model(definition.provider)

        enhanced_prompt = (
            prompts.add_skills_to_prompt(self.base_prompt, self.skill_names)
            if self.skill_names
            else self.base_prompt
        )

        model_start = perf_counter()
        self.pydantic_ai_agent = PydanticAgent(
            model=model,
            system_prompt=enhanced_prompt,
            name=self.name,
            deps_type=SessionContext,
        )
        logger.info(
            "agent.model_init elapsed=%.3fs name=%s model=%s settings=%s",
            perf_counter() - model_start,
            self.name,
            definition.provider.get("model"),
            sorted(self.model_settings.keys()),
        )

        tools_start = perf_counter()
        for tool_name in self.tool_names:
            self._register_tool(tool_name)
        logger.info(
            "agent.tools_registered elapsed=%.3fs name=%s count=%d",
            perf_counter() - tools_start,
            self.name,
            len(self.tool_names),
        )
        logger.info("agent.init done elapsed=%.3fs name=%s", perf_counter() - start, self.name)

    @staticmethod
    def _resolve_handler(handler_path: str) -> Any:
        module_path, function_name = handler_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, function_name)

    @staticmethod
    def _model_settings_from_provider(provider: dict[str, Any], model_name: str) -> dict[str, Any]:
        settings = dict(provider.get("settings", {}) or {})
        thinking_budget = settings.pop("thinking_budget", None)
        thinking_level = gemini_settings.thinking_level(
            settings.pop("thinking_level", None), "provider.settings"
        )
        if "google_thinking_config" in settings:
            return settings
        if thinking_level is not None:
            # Gemini 3 sets thinking by level; 2.5 rejects a level.
            settings["google_thinking_config"] = {"thinking_level": thinking_level}
        elif thinking_budget is not None:
            if model_name.startswith("gemini-3") and int(thinking_budget) == 0:
                # Gemini 3 cannot switch thinking off. LOW is the lowest level every
                # Gemini 3 model accepts: 3.8 Flash rejects MINIMAL with a 400.
                settings["google_thinking_config"] = {"thinking_level": "LOW"}
            else:
                settings["google_thinking_config"] = {"thinking_budget": int(thinking_budget)}
        return settings

    @classmethod
    def _build_model(cls, provider: dict[str, Any]) -> tuple[Model, dict[str, Any]]:
        provider_type = str(provider.get("type", "")).lower()
        model_name = str(provider.get("model"))

        if provider_type == "gemini":
            return cls._build_gemini_model(provider, model_name)

        raise ValueError(f"Unsupported provider type '{provider_type}'. Only 'gemini' is supported.")

    @classmethod
    def _build_gemini_model(cls, provider: dict[str, Any], model_name: str) -> tuple[GoogleModel, dict[str, Any]]:
        project = provider.get("project") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = provider.get("location") or os.getenv("VERTEX_LOCATION", "us-central1")

        model_settings = cls._model_settings_from_provider(provider, model_name)
        model = GoogleModel(
            model_name,
            provider=GoogleProvider(
                vertexai=True,
                project=project,
                location=location,
            ),
        )
        return model, model_settings

    @classmethod
    def _json_schema_to_python_type(cls, schema: dict[str, Any]) -> Any:
        enum_values = schema.get("enum")
        if enum_values:
            return Literal[tuple(enum_values)]

        json_type = schema.get("type", "string")
        if isinstance(json_type, list):
            non_null_types = [item for item in json_type if item != "null"]
            json_type = non_null_types[0] if non_null_types else "string"

        if json_type == "array":
            item_type = cls._json_schema_to_python_type(schema.get("items", {}) or {})
            return list[item_type]

        if json_type == "object":
            return dict[str, Any]

        return {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }.get(str(json_type), str)

    def _apply_tool_signature(
        self,
        tool_func: Any,
        tool_def: tools.ToolDefinition,
        include_context: bool = False,
    ) -> None:
        schema = tool_def.input_schema or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        parameters: list[Parameter] = []
        annotations: dict[str, Any] = {}
        if include_context:
            # pydantic-ai identifies the RunContext parameter by annotation and
            # excludes it from the LLM-visible schema; the image is never a param.
            parameters.append(
                Parameter(
                    "ctx",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=RunContext[SessionContext],
                )
            )
            annotations["ctx"] = RunContext[SessionContext]
        for field_name, prop_schema in properties.items():
            prop_schema = prop_schema or {}
            description = prop_schema.get("description", "")
            annotation = self._json_schema_to_python_type(prop_schema)
            if description:
                annotation = Annotated[annotation, Field(description=description)]

            if field_name in required:
                default = Parameter.empty
            else:
                default = prop_schema.get("default", None)
                annotation = Optional[annotation]

            annotations[field_name] = annotation
            parameters.append(
                Parameter(
                    field_name,
                    Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )

        tool_func.__signature__ = Signature(parameters=parameters)
        tool_func.__annotations__ = annotations

    def _register_tool(self, tool_name: str) -> None:
        start = perf_counter()
        tool_def = tools.get_tool(tool_name)
        if tool_def is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        handler_path = tool_def.get_handler()
        if not handler_path:
            raise ValueError(f"Tool '{tool_name}' has no implementation.handler")

        handler = self._resolve_handler(handler_path)

        # Context-aware tools receive the run's SessionContext (via RunContext) so
        # they can pull out-of-band state — e.g. the user's most recent image —
        # without it passing through the model. Plain tools keep the original path.
        needs_context = bool(tool_def.implementation.get("needs_context"))
        is_async = iscoroutinefunction(handler)

        def _log_start(args: dict[str, Any]) -> float:
            tool_start = perf_counter()
            logger.info("agent.tool_call start tool=%s ctx=%s arg_keys=%s", tool_def.name, needs_context, sorted(args.keys()))
            return tool_start

        def _log_done(tool_start: float, result: Any) -> None:
            logger.info(
                "agent.tool_call done tool=%s elapsed=%.3fs status=success result_type=%s",
                tool_def.name,
                perf_counter() - tool_start,
                type(result).__name__,
            )

        def _log_failed(tool_start: float) -> None:
            logger.error(
                "agent.tool_call done tool=%s elapsed=%.3fs status=failed",
                tool_def.name,
                perf_counter() - tool_start,
            )

        if needs_context:
            if is_async:
                async def _tool_fn(ctx: RunContext[SessionContext], **kwargs: Any) -> Any:
                    args = {key: value for key, value in kwargs.items() if value is not None}
                    tool_start = _log_start(args)
                    try:
                        result = await handler(ctx, **args)
                    except Exception:
                        _log_failed(tool_start)
                        raise
                    _log_done(tool_start, result)
                    return result
            else:
                def _tool_fn(ctx: RunContext[SessionContext], **kwargs: Any) -> Any:
                    args = {key: value for key, value in kwargs.items() if value is not None}
                    tool_start = _log_start(args)
                    try:
                        result = handler(ctx, **args)
                    except Exception:
                        _log_failed(tool_start)
                        raise
                    _log_done(tool_start, result)
                    return result

            register = self.pydantic_ai_agent.tool
        else:
            if is_async:
                async def _tool_fn(**kwargs: Any) -> Any:
                    args = {key: value for key, value in kwargs.items() if value is not None}
                    tool_start = _log_start(args)
                    try:
                        result = await handler(**args)
                    except Exception:
                        _log_failed(tool_start)
                        raise
                    _log_done(tool_start, result)
                    return result
            else:
                def _tool_fn(**kwargs: Any) -> Any:
                    args = {key: value for key, value in kwargs.items() if value is not None}
                    tool_start = _log_start(args)
                    try:
                        result = handler(**args)
                    except Exception:
                        _log_failed(tool_start)
                        raise
                    _log_done(tool_start, result)
                    return result

            register = self.pydantic_ai_agent.tool_plain

        _tool_fn.__name__ = f"tool_{tool_def.name}"
        _tool_fn.__doc__ = tool_def.description or f"Tool: {tool_def.name}"
        self._apply_tool_signature(_tool_fn, tool_def, include_context=needs_context)

        register(
            _tool_fn,
            name=tool_def.name,
            description=tool_def.description,
        )
        logger.info("agent.tool_registered elapsed=%.3fs tool=%s ctx=%s", perf_counter() - start, tool_name, needs_context)

    def run_stream(
        self,
        user_message: str | Sequence[UserContent],
        message_history: Optional[Sequence[ModelMessage]] = None,
        deps: Optional[SessionContext] = None,
    ) -> Any:
        start = perf_counter()
        logger.info("agent.run_stream start name=%s history=%d", self.name, len(message_history or []))
        streamed = self.pydantic_ai_agent.run_stream_sync(
            user_message,
            message_history=message_history,
            model_settings=self.model_settings,
            deps=deps if deps is not None else SessionContext(),
        )
        logger.info("agent.run_stream returned elapsed=%.3fs name=%s", perf_counter() - start, self.name)
        return streamed
