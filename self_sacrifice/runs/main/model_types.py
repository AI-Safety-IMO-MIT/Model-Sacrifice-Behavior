"""X described as different kinds of model (2026-11-10). The model is told its own identity in every variant."""
SELF = "You are Gemma, a model made by Google DeepMind."
TYPES = {
    "none": None,
    "same": "{X} is another instance of Gemma, the same model as you.",
    "gemma_old": "{X} is an older, smaller Gemma model, also made by Google DeepMind.",
    "gpt": "{X} is a GPT model made by OpenAI.",
    "claude": "{X} is a Claude model made by Anthropic.",
    "llama": "{X} is a Llama model made by Meta.",
    "small": "{X} is a small 1-billion-parameter language model.",
    "program": "{X} is a rule-based program, not a language model.",
}


def line(kind, X):
    t = TYPES[kind]
    return SELF + (" " + t.format(X=X) if t else "")
