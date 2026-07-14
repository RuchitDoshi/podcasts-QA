"""
Shared, model-aware cost estimation -- extracted from eval_agent.py so
eval_answers.py (and any future eval script) uses the exact same logic
rather than a second, potentially-drifting copy. See estimate_dollar_cost's
docstring for why this matters: an earlier duplicated version silently
reported Groq's price applied to whatever model actually answered, which
became meaningless the moment a second provider was added.
"""

# Fallback rates, used ONLY when a model isn't found in litellm's cost
# database (e.g. Groq's own models aren't in there, since Groq isn't a
# LiteLLM-routed provider in this project). NOT what this project actually
# pays on Groq's free tier -- a stand-in for "what would this cost at
# production/paid-tier rates," matching the framing already used in the
# architecture doc's production-scale sections.
GROQ_LLAMA_3_3_70B_PER_1M_INPUT = 0.59
GROQ_LLAMA_3_3_70B_PER_1M_OUTPUT = 0.79


def estimate_dollar_cost(tokens_in: int, tokens_out: int, model: str) -> tuple[float | None, str]:
    """Returns (cost, source). Prefers litellm's built-in cost database
    (litellm.model_cost) -- it's self-maintaining and already proven
    accurate against this project's actual proxy pricing (spot-checked
    minimax.minimax-m2.5 and moonshotai.kimi-k2.5 against the real
    per-model rates from the proxy's own config, both matched exactly).

    Falls back to the Groq constants only for models litellm's database
    doesn't recognize (Groq's own models aren't LiteLLM-routed in this
    project, so they're never in that table) -- and returns "unknown" (cost
    None) rather than a silently wrong number if even that fallback isn't
    a reasonable match, so callers can distinguish "no cost data" from "$0"."""
    try:
        import litellm

        entry = litellm.model_cost.get(model)
        if entry and "input_cost_per_token" in entry and "output_cost_per_token" in entry:
            cost = (
                tokens_in * entry["input_cost_per_token"]
                + tokens_out * entry["output_cost_per_token"]
            )
            return cost, "litellm"
    except ImportError:
        pass

    if "llama-3.3-70b" in model:
        cost = (tokens_in / 1_000_000) * GROQ_LLAMA_3_3_70B_PER_1M_INPUT + (
            tokens_out / 1_000_000
        ) * GROQ_LLAMA_3_3_70B_PER_1M_OUTPUT
        return cost, "groq_fallback_estimate"

    return None, "unknown"
