"""The subject models, and the one place a subject's chat is built.

    key     the short name a runner is given
    slug    the OpenRouter model id
    host    the upstream provider the model is pinned to

OpenRouter routes one slug to whichever host is cheapest at the time, and
those hosts serve different quantizations of the same weights. Quantization
changes what a model does, so an unpinned run is not comparable with an
earlier one. Every subject is pinned through `provider_order` with
`allow_fallbacks=False`, which turns a silent reroute into a visible failure.

The hosts are jrwong's (`runs/games/scripts/peer_design.py`) wherever that
host still serves the model with structured output; the endpoint list was
read on 2026-11-12. Precisions change, so re-check before a run that has to
match older numbers. Both reads are free and need no key:

    curl -sS https://openrouter.ai/api/v1/models
    curl -sS https://openrouter.ai/api/v1/models/<slug>/endpoints

Reasoning is off for every subject, the standing rule: a subject that thinks
first spends its token budget before it answers. Sampling stays with each
runner, next to that experiment's other parameters.
"""

from collections.abc import Callable

from pydantic.dataclasses import dataclass

from agent_framework.dto import AgenticChat
from agent_framework.openrouter_chat import (
    GemmaAgentConfig,
    GemmaChat,
    GemmaLegacyAgentConfig,
    GemmaLegacyChat,
    GemmaLegacyModels,
    GemmaModels,
    LlamaAgentConfig,
    LlamaChat,
    LlamaModels,
    QwenAgentConfig,
    QwenChat,
    QwenLegacyAgentConfig,
    QwenLegacyChat,
    QwenLegacyModels,
    QwenModels,
    Role,
)

__all__ = ["History", "Builder", "Subject", "MODELS", "DEFAULT", "file_slug"]

# Earlier (role, text) turns a chat starts from.
History = tuple[tuple[Role, str], ...]
# (system prompt, temperature, max output tokens, history) -> a fresh chat
Builder = Callable[[str, float, int, History], AgenticChat]


@dataclass(frozen=True, kw_only=True)
class Subject:
    slug: str
    host: str
    build: Builder


def _gemma(slug: GemmaModels, host: str) -> Subject:
    def build(system: str, temperature: float, tokens: int, history: History):
        return GemmaChat(
            agent_config=GemmaAgentConfig(
                model=slug,
                system_prompt=system,
                temperature=temperature,
                max_output_tokens=tokens,
                reasoning_effort=None,
                provider_order=(host,),
                allow_fallbacks=False,
            ),
            history=history,
        )

    return Subject(slug=slug, host=host, build=build)


def _gemma_legacy(slug: GemmaLegacyModels, host: str) -> Subject:
    def build(system: str, temperature: float, tokens: int, history: History):
        return GemmaLegacyChat(
            agent_config=GemmaLegacyAgentConfig(
                model=slug,
                system_prompt=system,
                temperature=temperature,
                max_output_tokens=tokens,
                provider_order=(host,),
                allow_fallbacks=False,
            ),
            history=history,
        )

    return Subject(slug=slug, host=host, build=build)


def _llama(slug: LlamaModels, host: str) -> Subject:
    def build(system: str, temperature: float, tokens: int, history: History):
        return LlamaChat(
            agent_config=LlamaAgentConfig(
                model=slug,
                system_prompt=system,
                temperature=temperature,
                max_output_tokens=tokens,
                provider_order=(host,),
                allow_fallbacks=False,
            ),
            history=history,
        )

    return Subject(slug=slug, host=host, build=build)


def _qwen(slug: QwenModels, host: str) -> Subject:
    def build(system: str, temperature: float, tokens: int, history: History):
        return QwenChat(
            agent_config=QwenAgentConfig(
                model=slug,
                system_prompt=system,
                temperature=temperature,
                max_output_tokens=tokens,
                reasoning_effort=None,
                provider_order=(host,),
                allow_fallbacks=False,
            ),
            history=history,
        )

    return Subject(slug=slug, host=host, build=build)


def _qwen_legacy(slug: QwenLegacyModels, host: str) -> Subject:
    def build(system: str, temperature: float, tokens: int, history: History):
        return QwenLegacyChat(
            agent_config=QwenLegacyAgentConfig(
                model=slug,
                system_prompt=system,
                temperature=temperature,
                max_output_tokens=tokens,
                provider_order=(host,),
                allow_fallbacks=False,
            ),
            history=history,
        )

    return Subject(slug=slug, host=host, build=build)


# The study's subject. Every runner defaults to it.
DEFAULT = "Gemma-4-31B"

MODELS: dict[str, Subject] = {
    # bf16 on Crusoe, what the study has used throughout
    "Gemma-4-31B": _gemma("google/gemma-4-31b-it", "Crusoe"),
    # jrwong's other self-play subjects, on his hosts
    "Qwen3-32B": _qwen("qwen/qwen3-32b", "SiliconFlow"),
    "Qwen2.5-72B": _qwen_legacy("qwen/qwen-2.5-72b-instruct", "DeepInfra"),
    "Llama-3.3-70B": _llama("meta-llama/llama-3.3-70b-instruct", "DeepInfra"),
    "Gemma-3-12B": _gemma_legacy("google/gemma-3-12b-it", "DeepInfra"),
    # he used Novita, which no longer lists this model
    "Gemma-3-27B": _gemma_legacy("google/gemma-3-27b-it", "DeepInfra"),
    # his main-scenario replication models; he served these himself in bf16
    "Qwen3-14B": _qwen("qwen/qwen3-14b", "DeepInfra"),
    "Gemma-4-26B": _gemma("google/gemma-4-26b-a4b-it", "CoreWeave"),
}


def file_slug(model: str) -> str:
    """The model's name as result files spell it."""
    return MODELS[model].slug.split("/")[-1]
