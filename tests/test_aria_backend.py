from unittest.mock import patch, MagicMock
import torch
from tt_midi_maker.generation.aria_backend import (
    load_model, generate_tokens, ARIA_MODELS,
)


def test_aria_models_list_non_empty():
    assert len(ARIA_MODELS) >= 1


def test_load_model_cpu_returns_model_and_label():
    mock_model = MagicMock()
    mock_tok   = MagicMock()
    with patch("tt_midi_maker.generation.aria_backend.AutoModelForCausalLM") as MockM, \
         patch("tt_midi_maker.generation.aria_backend.AutoTokenizer") as MockT:
        MockM.from_pretrained.return_value = mock_model
        MockT.from_pretrained.return_value = mock_tok
        model, tokenizer, label = load_model(ARIA_MODELS[0], device_ids=[])
    assert model is mock_model
    assert label == "cpu-fallback"


def test_load_model_tries_forge_with_devices():
    mock_model = MagicMock()
    mock_tok   = MagicMock()
    mock_compiled = MagicMock()
    with patch("tt_midi_maker.generation.aria_backend.AutoModelForCausalLM") as MockM, \
         patch("tt_midi_maker.generation.aria_backend.AutoTokenizer") as MockT, \
         patch("tt_midi_maker.generation.aria_backend._try_forge_compile",
               return_value=(mock_compiled, "tt-forge/2x")) as mock_forge:
        MockM.from_pretrained.return_value = mock_model
        MockT.from_pretrained.return_value = mock_tok
        model, tokenizer, label = load_model(ARIA_MODELS[0], device_ids=[0, 1])
    mock_forge.assert_called_once()
    assert label == "tt-forge/2x"


def test_generate_tokens_returns_list():
    mock_model = MagicMock()
    fake_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    mock_model.generate.return_value = fake_output
    result = generate_tokens(mock_model, input_tokens=[1, 2, 3], max_new_tokens=5)
    assert isinstance(result, list)
    assert result == [4, 5, 6, 7, 8]   # tokens after the 3 input tokens


def test_generate_tokens_passes_temperature():
    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 99]])
    generate_tokens(mock_model, input_tokens=[1, 2, 3],
                    max_new_tokens=1, temperature=0.5)
    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs["temperature"] == 0.5
