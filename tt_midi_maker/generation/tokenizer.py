"""
MidiTok REMI tokenizer wrapper. Aria was trained with REMI+ tokenization.

Uses miditok.REMI with REMI+ compatible config. The Aria tokenizer was not
present in the installed miditok version.

Encoding API: tok(midi_file_path) -> TokSequence
Decoding API: tok.tokens_to_midi(TokSequence) -> symusic Score -> .dump_midi(path)
"""
from pathlib import Path
import miditok
from miditok import TokenizerConfig
from miditok.classes import TokSequence

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_programs=True,
        use_tempo=True,
    )
    if hasattr(miditok, "Aria"):
        _tokenizer = miditok.Aria(config)
    else:
        _tokenizer = miditok.REMI(config)
    return _tokenizer


def encode_midi_file(midi_path: Path) -> list[int]:
    """Tokenize a MIDI file into a flat list of integer token IDs."""
    tok = get_tokenizer()
    result = tok(str(midi_path))
    if isinstance(result, TokSequence):
        return result.ids
    if isinstance(result, list):
        ids: list[int] = []
        for seq in result:
            ids.extend(seq.ids if hasattr(seq, "ids") else seq)
        return ids
    return list(result)


def decode_tokens_to_midi(tokens: list[int], output_path: Path) -> Path:
    """Decode a token ID list back to a MIDI file."""
    tok = get_tokenizer()
    seq = TokSequence(ids=tokens)
    score = tok.decode(seq)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(output_path))
    return output_path
