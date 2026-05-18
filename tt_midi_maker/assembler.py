from pathlib import Path
import mido
from .models.track import RoleTrack

TICKS_PER_BEAT = 480


def bpm_to_tempo(bpm: int) -> int:
    return int(60_000_000 / bpm)


def build_midi_file(
    role_tracks: list[RoleTrack],
    bpm: int,
    output_path: Path,
    ticks_per_beat: int = TICKS_PER_BEAT,
) -> Path:
    """Assemble a Type-1 multi-track MIDI file from RoleTracks."""
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=bpm_to_tempo(bpm), time=0))
    mid.tracks.append(tempo_track)

    for role_track in role_tracks:
        track = mido.MidiTrack()
        track.name = role_track.role

        ch = role_track.channel - 1   # mido is 0-indexed

        if role_track.channel != 10:
            track.append(mido.Message("program_change", channel=ch,
                                      program=role_track.program, time=0))

        events: list[tuple[int, str, int, int]] = []
        for note in role_track.notes:
            events.append((note.start_tick, "note_on",  note.pitch, note.velocity))
            events.append((note.start_tick + note.duration_ticks, "note_off", note.pitch, 0))

        events.sort(key=lambda e: e[0])
        current_tick = 0
        for abs_tick, msg_type, pitch, vel in events:
            delta = abs_tick - current_tick
            track.append(mido.Message(msg_type, channel=ch, note=pitch,
                                      velocity=vel, time=delta))
            current_tick = abs_tick

        mid.tracks.append(track)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))
    return output_path
