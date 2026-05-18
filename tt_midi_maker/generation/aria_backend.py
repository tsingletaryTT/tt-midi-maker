import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ordered fallback chain — first one that compiles wins
ARIA_MODELS = [
    "nlp4music/aria-medium",
    "nlp4music/aria-mini",
    "skytnt/midi-model",
]

_loaded: tuple | None = None   # (model, tokenizer, label)


def _try_forge_compile(model, device_ids: list[int]):
    """Attempt tt-forge compilation. Returns (compiled_model, label) or raises."""
    import forge
    sample = torch.zeros((1, 16), dtype=torch.long)
    compiled = forge.compile(model, sample, module_name="aria_midi")
    return compiled, f"tt-forge/{len(device_ids)}x"


def load_model(model_name: str, device_ids: list[int]) -> tuple:
    """Load model. Returns (model, tokenizer, hardware_label)."""
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device_ids:
        try:
            compiled, label = _try_forge_compile(model, device_ids)
            return compiled, tokenizer, label
        except Exception as e:
            print(f"[tt-midi-maker] tt-forge compile failed ({e}), falling back to CPU")

    return model, tokenizer, "cpu-fallback"


def get_model(device_ids: list[int] | None = None) -> tuple:
    """Lazily load model, trying ARIA_MODELS in order."""
    global _loaded
    if _loaded is not None:
        return _loaded
    devices = device_ids or []
    for name in ARIA_MODELS:
        try:
            result = load_model(name, devices)
            _loaded = result
            print(f"[tt-midi-maker] loaded {name} ({result[2]})")
            return result
        except Exception as e:
            print(f"[tt-midi-maker] could not load {name}: {e}")
    raise RuntimeError(f"No MIDI model could be loaded from {ARIA_MODELS}")


def generate_tokens(
    model,
    input_tokens: list[int],
    max_new_tokens: int = 512,
    temperature: float = 0.9,
) -> list[int]:
    """Run model.generate and return only the newly generated token IDs."""
    input_tensor = torch.tensor([input_tokens])
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=0,
        )
    return output[0][len(input_tokens):].tolist()
