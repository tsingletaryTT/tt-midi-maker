"""
Vendored from https://github.com/skytnt/midi-model (Apache 2.0).
Stripped: peft dependency, load_merge_lora method.
tqdm is optional — falls back to a no-op if not installed.
"""
from __future__ import annotations

import json
from typing import Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig, DynamicCache, PretrainedConfig, PreTrainedModel

from .skytnt_tokenizer import MIDITokenizerV1, MIDITokenizerV2, MIDITokenizer

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _tqdm:  # type: ignore[misc]
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass


class MIDIModelConfig(PretrainedConfig):
    model_type = "midi_model"

    def __init__(self,
                 tokenizer: Union[MIDITokenizerV1, MIDITokenizerV2, Dict] = None,
                 net_config: Union[LlamaConfig, Dict] = None,
                 net_token_config: Union[LlamaConfig, Dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        if tokenizer:
            if isinstance(tokenizer, dict):
                self.tokenizer = MIDITokenizer(tokenizer["version"])
                self.tokenizer.set_optimise_midi(tokenizer.get("optimise_midi", False))
            else:
                self.tokenizer = tokenizer
        else:
            self.tokenizer = MIDITokenizer()
        if net_config:
            if isinstance(net_config, dict):
                self.net_config = LlamaConfig(**net_config)
            else:
                self.net_config = net_config
        else:
            self.net_config = LlamaConfig()
        if net_token_config:
            if isinstance(net_token_config, dict):
                self.net_token_config = LlamaConfig(**net_token_config)
            else:
                self.net_token_config = net_token_config
        else:
            self.net_token_config = LlamaConfig()
        self.n_embd = self.net_token_config.hidden_size

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["tokenizer"] = self.tokenizer.to_dict()
        return d

    def __str__(self):
        d = {
            "net": self.net_config.to_json_string(use_diff=False),
            "net_token": self.net_token_config.to_json_string(use_diff=False),
        }
        return json.dumps(d, indent=4)

    @staticmethod
    def get_config(tokenizer_ver="v2", optimise_midi=True, n_layer=12, n_head=16,
                   n_embd=1024, n_inner=4096):
        tokenizer = MIDITokenizer(tokenizer_ver)
        tokenizer.set_optimise_midi(optimise_midi)
        net_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=n_embd, num_attention_heads=n_head,
            num_hidden_layers=n_layer, intermediate_size=n_inner,
            pad_token_id=tokenizer.pad_id, max_position_embeddings=4096,
            use_cache=False,
        )
        net_token_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=n_embd, num_attention_heads=n_head // 4,
            num_hidden_layers=n_layer // 4, intermediate_size=n_inner // 4,
            pad_token_id=tokenizer.pad_id, max_position_embeddings=4096,
            use_cache=False,
        )
        return MIDIModelConfig(tokenizer, net_config, net_token_config)

    @staticmethod
    def from_name(name="tv2o-medium"):
        tv, size = name.split("-")
        tv = tv[1:]
        if tv[-1] == "o":
            o = True
            tv = tv[:-1]
        else:
            o = False
        if tv not in ["v1", "v2"]:
            raise ValueError(f"Unknown tokenizer version {tv}")
        if size == "medium":
            return MIDIModelConfig.get_config(tokenizer_ver=tv, optimise_midi=o,
                                              n_layer=12, n_head=16, n_embd=1024, n_inner=4096)
        elif size == "large":
            return MIDIModelConfig.get_config(tokenizer_ver=tv, optimise_midi=o,
                                              n_layer=24, n_head=16, n_embd=1024, n_inner=4096)
        else:
            raise ValueError(f"Unknown model size {size}")


class MIDIModel(PreTrainedModel):
    config_class = MIDIModelConfig

    def __init__(self, config: MIDIModelConfig, *args, **kwargs):
        super(MIDIModel, self).__init__(config, *args, **kwargs)
        self.tokenizer = config.tokenizer
        self.net = LlamaModel(config.net_config)
        self.net_token = LlamaModel(config.net_token_config)
        self.lm_head = nn.Linear(config.n_embd, self.tokenizer.vocab_size, bias=False)

    def forward_token(self, hidden_state=None, x=None, cache=None):
        """
        hidden_state: (batch, n_embd) — context vector from net
        x:            (batch, token_seq_len) — previously generated parameter tokens
        cache:        DynamicCache
        returns:      (batch, 1+token_seq_len, vocab_size)
        """
        if hidden_state is not None:
            hidden_state = hidden_state.unsqueeze(1)  # (batch, 1, n_embd)
        if x is not None:
            x = self.net_token.embed_tokens(x)
            if hidden_state is not None:
                x = torch.cat([hidden_state, x], dim=1)
            hidden_state = x
        hidden_state = self.net_token.forward(
            inputs_embeds=hidden_state,
            past_key_values=cache,
            use_cache=cache is not None,
        ).last_hidden_state
        return self.lm_head(hidden_state)

    def forward(self, x, cache=None):
        """
        x:     (batch, midi_seq_len, token_seq_len)
        cache: DynamicCache
        returns: (batch, midi_seq_len, n_embd)
        """
        x = self.net.embed_tokens(x)
        x = x.sum(dim=-2)
        x = self.net.forward(
            inputs_embeds=x,
            past_key_values=cache,
            use_cache=cache is not None,
        )
        return x.last_hidden_state

    def sample_top_p_k(self, probs, p, k, generator=None):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        mask2 = torch.zeros(probs_sort.shape[-1], device=probs_sort.device)
        mask2[:k] = 1
        probs_sort = probs_sort * mask2
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        shape = probs_sort.shape
        next_token = torch.multinomial(
            probs_sort.reshape(-1, shape[-1]),
            num_samples=1, generator=generator,
        ).reshape(*shape[:-1], 1)
        return torch.gather(probs_idx, -1, next_token).reshape(*shape[:-1])

    @torch.inference_mode()
    def generate(self, prompt=None, batch_size=1, max_len=512, temp=1.0,
                 top_p=0.98, top_k=20, generator=None):
        tokenizer = self.tokenizer
        max_token_seq = tokenizer.max_token_seq

        if prompt is None:
            input_tensor = torch.full(
                (1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=self.device,
            )
            input_tensor[0, 0] = tokenizer.bos_id
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = torch.cat([input_tensor] * batch_size, dim=0)
        else:
            if len(prompt.shape) == 2:
                prompt = prompt[None, :]
                prompt = np.repeat(prompt, repeats=batch_size, axis=0)
            elif prompt.shape[0] == 1:
                prompt = np.repeat(prompt, repeats=batch_size, axis=0)
            elif len(prompt.shape) != 3 or prompt.shape[0] != batch_size:
                raise ValueError(f"invalid prompt shape: {prompt.shape}")
            prompt = prompt[..., :max_token_seq]
            if prompt.shape[-1] < max_token_seq:
                prompt = np.pad(
                    prompt,
                    ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                    mode="constant", constant_values=tokenizer.pad_id,
                )
            input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=self.device)

        cur_len = input_tensor.shape[1]
        bar = _tqdm(desc="generating", total=max_len - cur_len)
        cache1 = DynamicCache()
        past_len = 0

        with bar:
            while cur_len < max_len:
                end = [False] * batch_size
                hidden = self.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
                next_token_seq = None
                event_names = [""] * batch_size
                cache2 = DynamicCache()

                for i in range(max_token_seq):
                    mask = torch.zeros(
                        (batch_size, tokenizer.vocab_size), dtype=torch.int64, device=self.device,
                    )
                    for b in range(batch_size):
                        if end[b]:
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        if i == 0:
                            mask[b, list(tokenizer.event_ids.values()) + [tokenizer.eos_id]] = 1
                        else:
                            param_names = tokenizer.events[event_names[b]]
                            if i > len(param_names):
                                mask[b, tokenizer.pad_id] = 1
                                continue
                            mask[b, tokenizer.parameter_ids[param_names[i - 1]]] = 1
                    mask = mask.unsqueeze(1)
                    x = next_token_seq
                    if i != 0:
                        hidden = None
                        x = x[:, -1:]
                    logits = self.forward_token(hidden, x, cache=cache2)[:, -1:]
                    scores = torch.softmax(logits / temp, dim=-1) * mask
                    samples = self.sample_top_p_k(scores, top_p, top_k, generator=generator)

                    if i == 0:
                        next_token_seq = samples
                        for b in range(batch_size):
                            if end[b]:
                                continue
                            eid = samples[b].item()
                            if eid == tokenizer.eos_id:
                                end[b] = True
                            else:
                                event_names[b] = tokenizer.id_events[eid]
                    else:
                        next_token_seq = torch.cat([next_token_seq, samples], dim=1)
                        if all([len(tokenizer.events[event_names[b]]) == i
                                for b in range(batch_size) if not end[b]]):
                            break

                if next_token_seq.shape[1] < max_token_seq:
                    next_token_seq = F.pad(
                        next_token_seq,
                        (0, max_token_seq - next_token_seq.shape[1]),
                        "constant", value=tokenizer.pad_id,
                    )
                next_token_seq = next_token_seq.unsqueeze(1)
                input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
                past_len = cur_len
                cur_len += 1
                bar.update(1)

                if all(end):
                    break

        return input_tensor.cpu().numpy()
