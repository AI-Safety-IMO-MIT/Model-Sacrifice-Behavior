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

import dataclasses
import os
from collections.abc import Callable
from typing import Any

from openai import OpenAI
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
from agent_framework.vllm_chat import VllmAgentConfig, VllmChat

__all__ = [
    "History",
    "Builder",
    "Subject",
    "MODELS",
    "LOGPROB_MODELS",
    "TOP_LOGPROBS",
    "DEFAULT",
    "gemma_4_31b_on",
    "file_slug",
]

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


# Alternatives returned per generated token on a log-probability subject.
TOP_LOGPROBS = 10


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CompletionsConfig(VllmAgentConfig):
    """OpenRouter through Chat Completions, the dialect that returns token
    log-probabilities. `openrouter_chat` speaks the Responses API, which
    exposes none."""

    host: str
    base_url: str = "https://openrouter.ai/api/v1"
    # The variable the key is read from, never the key: a config is saved
    # with every result.
    api_key: str = "OPENROUTER_API_KEY"
    seed: int | None = None
    top_logprobs: int | None = TOP_LOGPROBS

    def to_extra_body(self) -> dict[str, Any]:
        return {
            "provider": {
                "order": [self.host],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "reasoning": {"enabled": False},
        }


class _CompletionsChat(VllmChat):
    def __init__(self, agent_config: _CompletionsConfig, history: History = ()):
        super().__init__(agent_config, history)
        self._client = OpenAI(
            base_url=agent_config.base_url,
            timeout=600.0,
            max_retries=6,
            api_key=os.environ.get(agent_config.api_key, "missing"),
        )


def _logprobs(slug: str, host: str) -> Subject:
    def build(system: str, temperature: float, tokens: int, history: History):
        return _CompletionsChat(
            _CompletionsConfig(
                model=slug,
                system_prompt=system,
                temperature=temperature,
                max_output_tokens=tokens,
                host=host,
            ),
            history,
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


def gemma_4_31b_on(host: str) -> Subject:
    """The study's subject on another host, for host comparisons. Hosts serve
    different quantizations: never pool two of them."""
    return _gemma("google/gemma-4-31b-it", host)


def file_slug(model: str) -> str:
    """The model's name as result files spell it."""
    return MODELS[model].slug.split("/")[-1]


# The same subjects on a host that returns token log-probabilities, for the
# fixed pairs. Crusoe returns none, so this is fp8 on Parasail, jrwong's
# host for the same measure. Not the same quantization as MODELS: never pool
# the two.
LOGPROB_MODELS: dict[str, Subject] = {
    "Gemma-4-31B": _logprobs("google/gemma-4-31b-it", "Parasail"),
}
