"""
Vendored from https://github.com/skytnt/midi-model (Apache 2.0).
Stripped: PIL.Image import, midi2img method.
"""
from __future__ import annotations

import random
from typing import Dict, Any

import numpy as np


class MIDITokenizerV1:
    def __init__(self):
        self.version = "v1"
        self.optimise_midi = False
        self.vocab_size = 0

        def allocate_ids(size):
            ids = [self.vocab_size + i for i in range(size)]
            self.vocab_size += size
            return ids

        self.pad_id = allocate_ids(1)[0]
        self.bos_id = allocate_ids(1)[0]
        self.eos_id = allocate_ids(1)[0]
        self.events = {
            "note": ["time1", "time2", "track", "duration", "channel", "pitch", "velocity"],
            "patch_change": ["time1", "time2", "track", "channel", "patch"],
            "control_change": ["time1", "time2", "track", "channel", "controller", "value"],
            "set_tempo": ["time1", "time2", "track", "bpm"],
        }
        self.event_parameters = {
            "time1": 128, "time2": 16, "duration": 2048, "track": 128, "channel": 16,
            "pitch": 128, "velocity": 128, "patch": 128, "controller": 128, "value": 128,
            "bpm": 256,
        }
        self.event_ids = {e: allocate_ids(1)[0] for e in self.events.keys()}
        self.id_events = {i: e for e, i in self.event_ids.items()}
        self.parameter_ids = {p: allocate_ids(s) for p, s in self.event_parameters.items()}
        self.max_token_seq = max([len(ps) for ps in self.events.values()]) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "optimise_midi": self.optimise_midi,
            "vocab_size": self.vocab_size,
            "events": self.events,
            "event_parameters": self.event_parameters,
            "max_token_seq": self.max_token_seq,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
        }

    def set_optimise_midi(self, optimise_midi=True):
        self.optimise_midi = optimise_midi

    @staticmethod
    def tempo2bpm(tempo):
        return 60 / (tempo / 10 ** 6)

    @staticmethod
    def bpm2tempo(bpm):
        if bpm == 0:
            bpm = 1
        return int((60 / bpm) * 10 ** 6)

    def tokenize(self, midi_score, add_bos_eos=True, cc_eps=4, tempo_eps=4,
                 remap_track_channel=None, add_default_instr=None, remove_empty_channels=None):
        if remap_track_channel is None:
            remap_track_channel = self.optimise_midi
        if add_default_instr is None:
            add_default_instr = self.optimise_midi
        if remove_empty_channels is None:
            remove_empty_channels = self.optimise_midi

        ticks_per_beat = midi_score[0]
        event_list = {}
        track_idx_map = {i: dict() for i in range(16)}
        track_idx_dict = {}
        channels = []
        patch_channels = []
        empty_channels = [True] * 16
        channel_note_tracks = {i: list() for i in range(16)}
        for track_idx, track in enumerate(midi_score[1:129]):
            last_notes = {}
            patch_dict = {}
            control_dict = {}
            last_tempo = 0
            for event in track:
                if event[0] not in self.events:
                    continue
                c = -1
                t = round(16 * event[1] / ticks_per_beat)
                new_event = [event[0], t // 16, t % 16, track_idx] + event[2:]
                if event[0] == "note":
                    c = event[3]
                    if c > 15 or c < 0:
                        continue
                    empty_channels[c] = False
                    track_idx_dict.setdefault(c, track_idx)
                    note_tracks = channel_note_tracks[c]
                    if track_idx not in note_tracks:
                        note_tracks.append(track_idx)
                    new_event[4] = max(1, round(16 * new_event[4] / ticks_per_beat))
                elif event[0] == "set_tempo":
                    if new_event[4] == 0:
                        continue
                    bpm = int(self.tempo2bpm(new_event[4]))
                    new_event[4] = min(bpm, 255)
                if event[0] == "note":
                    key = tuple(new_event[:4] + new_event[5:-1])
                else:
                    key = tuple(new_event[:-1])
                if event[0] == "patch_change":
                    c, p = event[2:]
                    if c > 15 or c < 0:
                        continue
                    last_p = patch_dict.setdefault(c, None)
                    if last_p == p:
                        continue
                    patch_dict[c] = p
                    if c not in patch_channels:
                        patch_channels.append(c)
                elif event[0] == "control_change":
                    c, cc, v = event[2:]
                    if c > 15 or c < 0:
                        continue
                    last_v = control_dict.setdefault((c, cc), 0)
                    if abs(last_v - v) < cc_eps:
                        continue
                    control_dict[(c, cc)] = v
                elif event[0] == "set_tempo":
                    tempo = new_event[-1]
                    if abs(last_tempo - tempo) < tempo_eps:
                        continue
                    last_tempo = tempo

                if c != -1:
                    if c not in channels:
                        channels.append(c)
                    tr_map = track_idx_map[c]
                    if track_idx not in tr_map:
                        tr_map[track_idx] = 0

                if event[0] == "note":
                    cp = tuple(new_event[5:7])
                    if cp in last_notes:
                        last_note_key, last_note = last_notes[cp]
                        last_t = last_note[1] * 16 + last_note[2]
                        last_note[4] = max(0, min(last_note[4], t - last_t))
                        if last_note[4] == 0:
                            event_list.pop(last_note_key)
                    last_notes[cp] = (key, new_event)
                event_list[key] = new_event
        event_list = list(event_list.values())
        empty_channels = [c for c in channels if empty_channels[c]]

        if remap_track_channel:
            patch_channels = []
            channels_count = 0
            channels_map = {9: 9} if 9 in channels else {}
            if remove_empty_channels:
                channels = sorted(channels, key=lambda x: 1 if x in empty_channels else 0)
            for c in channels:
                if c == 9:
                    continue
                channels_map[c] = channels_count
                channels_count += 1
                if channels_count == 9:
                    channels_count = 10
            channels = list(channels_map.values())
            track_count = 0
            track_idx_map_order = [k for k, v in sorted(list(channels_map.items()), key=lambda x: x[1])]
            for c in track_idx_map_order:
                if remove_empty_channels and c in empty_channels:
                    continue
                tr_map = track_idx_map[c]
                for track_idx in tr_map:
                    note_tracks = channel_note_tracks[c]
                    if len(note_tracks) != 0 and track_idx not in note_tracks:
                        continue
                    track_count += 1
                    tr_map[track_idx] = track_count
            for c in track_idx_map_order:
                if not (remove_empty_channels and c in empty_channels):
                    continue
                tr_map = track_idx_map[c]
                for track_idx in tr_map:
                    note_tracks = channel_note_tracks[c]
                    if not (len(note_tracks) != 0 and track_idx not in note_tracks):
                        continue
                    track_count += 1
                    tr_map[track_idx] = track_count

            empty_channels = [channels_map[c] for c in empty_channels]
            track_idx_dict = {}
            for event in event_list:
                name = event[0]
                track_idx = event[3]
                if name == "note":
                    c = event[5]
                    event[5] = channels_map[c]
                    event[3] = track_idx_map[c][track_idx]
                    track_idx_dict.setdefault(event[5], event[3])
                elif name == "set_tempo":
                    event[3] = 0
                elif name == "control_change" or name == "patch_change":
                    c = event[4]
                    event[4] = channels_map[c]
                    tr_map = track_idx_map[c]
                    note_tracks = channel_note_tracks[c]
                    if len(note_tracks) != 0 and track_idx not in note_tracks:
                        track_idx = channel_note_tracks[c][0]
                    new_track_idx = tr_map[track_idx]
                    event[3] = new_track_idx
                    if name == "patch_change" and event[4] not in patch_channels:
                        patch_channels.append(event[4])

        if add_default_instr:
            for c in channels:
                if c not in patch_channels and c in track_idx_dict:
                    event_list.append(["patch_change", 0, 0, track_idx_dict[c], c, 0])

        events_name_order = {"set_tempo": 0, "patch_change": 1, "control_change": 2, "note": 3}
        events_order = lambda e: e[1:4] + [events_name_order[e[0]]]
        event_list = sorted(event_list, key=events_order)

        setup_events = {}
        notes_in_setup = False
        for i, event in enumerate(event_list):
            new_event = [*event]
            if event[0] != "note":
                new_event[1] = 0
                new_event[2] = 0
            has_next = False
            has_pre = False
            if i < len(event_list) - 1:
                next_event = event_list[i + 1]
                has_next = event[1] + event[2] == next_event[1] + next_event[2]
            if notes_in_setup and i > 0:
                pre_event = event_list[i - 1]
                has_pre = event[1] + event[2] == pre_event[1] + pre_event[2]
            if (event[0] == "note" and not has_next) or (notes_in_setup and not has_pre):
                event_list = sorted(setup_events.values(), key=events_order) + event_list[i:]
                break
            else:
                if event[0] == "note":
                    notes_in_setup = True
                    key = tuple([event[0]] + event[3:-2])
                else:
                    key = tuple([event[0]] + event[3:-1])
            setup_events[key] = new_event

        last_t1 = 0
        midi_seq = []
        for event in event_list:
            if remove_empty_channels and event[0] in ["control_change", "patch_change"] and event[4] in empty_channels:
                continue
            cur_t1 = event[1]
            event[1] = event[1] - last_t1
            tokens = self.event2tokens(event)
            if not tokens:
                continue
            midi_seq.append(tokens)
            last_t1 = cur_t1

        if add_bos_eos:
            bos = [self.bos_id] + [self.pad_id] * (self.max_token_seq - 1)
            eos = [self.eos_id] + [self.pad_id] * (self.max_token_seq - 1)
            midi_seq = [bos] + midi_seq + [eos]
        return midi_seq

    def event2tokens(self, event):
        name = event[0]
        params = event[1:]
        if not all([0 <= params[i] < self.event_parameters[p]
                    for i, p in enumerate(self.events[name])]):
            return []
        tokens = [self.event_ids[name]] + [self.parameter_ids[p][params[i]]
                                           for i, p in enumerate(self.events[name])]
        tokens += [self.pad_id] * (self.max_token_seq - len(tokens))
        return tokens

    def tokens2event(self, tokens):
        if tokens[0] not in self.id_events:
            return []
        name = self.id_events[tokens[0]]
        if len(tokens) <= len(self.events[name]):
            return []
        params = tokens[1:]
        params = [params[i] - self.parameter_ids[p][0] for i, p in enumerate(self.events[name])]
        if not all([0 <= params[i] < self.event_parameters[p]
                    for i, p in enumerate(self.events[name])]):
            return []
        return [name] + params

    def detokenize(self, midi_seq):
        ticks_per_beat = 480
        tracks_dict = {}
        t1 = 0
        for tokens in midi_seq:
            if tokens[0] in self.id_events:
                event = self.tokens2event(tokens)
                if not event:
                    continue
                name = event[0]
                if name == "set_tempo":
                    event[4] = self.bpm2tempo(event[4])
                if event[0] == "note":
                    event[4] = int(event[4] * ticks_per_beat / 16)
                t1 += event[1]
                t = t1 * 16 + event[2]
                t = int(t * ticks_per_beat / 16)
                track_idx = event[3]
                if track_idx not in tracks_dict:
                    tracks_dict[track_idx] = []
                tracks_dict[track_idx].append([event[0], t] + event[4:])
        tracks = [tr for idx, tr in sorted(list(tracks_dict.items()), key=lambda it: it[0])]

        for i in range(len(tracks)):
            track = tracks[i]
            track = sorted(track, key=lambda e: e[1])
            last_note_t = {}
            zero_len_notes = []
            for e in reversed(track):
                if e[0] == "note":
                    t, d, c, p = e[1:5]
                    key = (c, p)
                    if key in last_note_t:
                        d = min(d, max(last_note_t[key] - t, 0))
                    last_note_t[key] = t
                    e[2] = d
                    if d == 0:
                        zero_len_notes.append(e)
            for e in zero_len_notes:
                track.remove(e)
            tracks[i] = track
        return [ticks_per_beat, *tracks]

    def augment(self, midi_seq, max_pitch_shift=4, max_vel_shift=10, max_cc_val_shift=10,
                max_bpm_shift=10, max_track_shift=0, max_channel_shift=16):
        pitch_shift = random.randint(-max_pitch_shift, max_pitch_shift)
        vel_shift = random.randint(-max_vel_shift, max_vel_shift)
        cc_val_shift = random.randint(-max_cc_val_shift, max_cc_val_shift)
        bpm_shift = random.randint(-max_bpm_shift, max_bpm_shift)
        track_shift = random.randint(0, max_track_shift)
        channel_shift = random.randint(0, max_channel_shift)
        midi_seq_new = []
        for tokens in midi_seq:
            tokens_new = [*tokens]
            if tokens[0] in self.id_events:
                name = self.id_events[tokens[0]]
                for i, pn in enumerate(self.events[name]):
                    if pn == "track":
                        tr = tokens[1 + i] - self.parameter_ids[pn][0]
                        tr = (tr + track_shift) % self.event_parameters[pn]
                        tokens_new[1 + i] = self.parameter_ids[pn][tr]
                    elif pn == "channel":
                        c = tokens[1 + i] - self.parameter_ids[pn][0]
                        c0 = c
                        c = (c + channel_shift) % self.event_parameters[pn]
                        if c0 == 9:
                            c = 9
                        elif c == 9:
                            c = (9 + channel_shift) % self.event_parameters[pn]
                        tokens_new[1 + i] = self.parameter_ids[pn][c]
                if name == "note":
                    c = tokens[5] - self.parameter_ids["channel"][0]
                    p = tokens[6] - self.parameter_ids["pitch"][0]
                    v = tokens[7] - self.parameter_ids["velocity"][0]
                    if c != 9:
                        p += pitch_shift
                    if not 0 <= p < 128:
                        return midi_seq
                    v = max(1, min(127, v + vel_shift))
                    tokens_new[6] = self.parameter_ids["pitch"][p]
                    tokens_new[7] = self.parameter_ids["velocity"][v]
                elif name == "control_change":
                    cc = tokens[5] - self.parameter_ids["controller"][0]
                    val = tokens[6] - self.parameter_ids["value"][0]
                    if cc in [1, 2, 7, 11]:
                        val = max(1, min(127, val + cc_val_shift))
                    tokens_new[6] = self.parameter_ids["value"][val]
                elif name == "set_tempo":
                    bpm = tokens[4] - self.parameter_ids["bpm"][0]
                    bpm = max(1, min(255, bpm + bpm_shift))
                    tokens_new[4] = self.parameter_ids["bpm"][bpm]
            midi_seq_new.append(tokens_new)
        return midi_seq_new


class MIDITokenizerV2:
    def __init__(self):
        self.version = "v2"
        self.optimise_midi = False
        self.vocab_size = 0

        def allocate_ids(size):
            ids = [self.vocab_size + i for i in range(size)]
            self.vocab_size += size
            return ids

        self.pad_id = allocate_ids(1)[0]
        self.bos_id = allocate_ids(1)[0]
        self.eos_id = allocate_ids(1)[0]
        self.events = {
            "note": ["time1", "time2", "track", "channel", "pitch", "velocity", "duration"],
            "patch_change": ["time1", "time2", "track", "channel", "patch"],
            "control_change": ["time1", "time2", "track", "channel", "controller", "value"],
            "set_tempo": ["time1", "time2", "track", "bpm"],
            "time_signature": ["time1", "time2", "track", "nn", "dd"],
            "key_signature": ["time1", "time2", "track", "sf", "mi"],
        }
        self.event_parameters = {
            "time1": 128, "time2": 16, "duration": 2048, "track": 128, "channel": 16,
            "pitch": 128, "velocity": 128, "patch": 128, "controller": 128, "value": 128,
            "bpm": 384, "nn": 16, "dd": 4, "sf": 15, "mi": 2,
        }
        self.event_ids = {e: allocate_ids(1)[0] for e in self.events.keys()}
        self.id_events = {i: e for e, i in self.event_ids.items()}
        self.parameter_ids = {p: allocate_ids(s) for p, s in self.event_parameters.items()}
        self.max_token_seq = max([len(ps) for ps in self.events.values()]) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "optimise_midi": self.optimise_midi,
            "vocab_size": self.vocab_size,
            "events": self.events,
            "event_parameters": self.event_parameters,
            "max_token_seq": self.max_token_seq,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
        }

    def set_optimise_midi(self, optimise_midi=True):
        self.optimise_midi = optimise_midi

    @staticmethod
    def tempo2bpm(tempo):
        return 60 / (tempo / 10 ** 6)

    @staticmethod
    def bpm2tempo(bpm):
        if bpm == 0:
            bpm = 1
        return int((60 / bpm) * 10 ** 6)

    @staticmethod
    def sf2key(sf):
        return (sf * 7) % 12

    @staticmethod
    def key2sf(k, mi):
        sf = (k * 7) % 12
        if sf > 6 or (mi == 1 and sf >= 5):
            sf -= 12
        return sf

    @staticmethod
    def detect_key_signature(key_hist, threshold=0.7):
        if len(key_hist) != 12:
            return None
        if sum(key_hist) == 0:
            return None
        p = sum(sorted(key_hist, reverse=True)[:7]) / sum(key_hist)
        if p < threshold:
            return None
        keys = [x[1] for x in sorted(zip(key_hist, range(len(key_hist))),
                                      reverse=True, key=lambda x: x[0])[:7]]
        keys = sorted(keys)
        semitones = []
        for i in range(len(keys)):
            dis = keys[i] - keys[i - 1]
            if dis == 1 or dis == -11:
                semitones.append(keys[i])
        if len(semitones) != 2:
            return None
        semitones_dis = semitones[1] - semitones[0]
        if semitones_dis == 5:
            root_key = semitones[0]
        elif semitones_dis == 7:
            root_key = semitones[1]
        else:
            return None
        return root_key

    def tokenize(self, midi_score, add_bos_eos=True, cc_eps=4, tempo_eps=4,
                 remap_track_channel=None, add_default_instr=None, remove_empty_channels=None):
        if remap_track_channel is None:
            remap_track_channel = self.optimise_midi
        if add_default_instr is None:
            add_default_instr = self.optimise_midi
        if remove_empty_channels is None:
            remove_empty_channels = self.optimise_midi

        ticks_per_beat = midi_score[0]
        event_list = {}
        track_idx_map = {i: dict() for i in range(16)}
        track_idx_dict = {}
        channels = []
        patch_channels = []
        empty_channels = [True] * 16
        channel_note_tracks = {i: list() for i in range(16)}
        note_key_hist = [0] * 12
        key_sigs = []
        track_to_channels = {}
        for track_idx, track in enumerate(midi_score[1:129]):
            last_notes = {}
            patch_dict = {}
            control_dict = {}
            last_bpm = 0
            track_channels = []
            track_to_channels.setdefault(track_idx, track_channels)
            for event in track:
                if event[0] not in self.events:
                    continue
                name = event[0]
                c = -1
                t = round(16 * event[1] / ticks_per_beat)
                new_event = [name, t // 16, t % 16, track_idx]
                if name == "note":
                    d, c, p, v = event[2:]
                    if not (0 <= c <= 15):
                        continue
                    d = max(1, round(16 * d / ticks_per_beat))
                    new_event += [c, p, v, d]
                    empty_channels[c] = False
                    track_idx_dict.setdefault(c, track_idx)
                    note_tracks = channel_note_tracks[c]
                    if track_idx not in note_tracks:
                        note_tracks.append(track_idx)
                    if c != 9:
                        note_key_hist[p % 12] += 1
                    if c not in track_channels:
                        track_channels.append(c)
                elif name == "patch_change":
                    c, p = event[2:]
                    if not (0 <= c <= 15):
                        continue
                    new_event += [c, p]
                    last_p = patch_dict.setdefault(c, None)
                    if last_p == p:
                        continue
                    patch_dict[c] = p
                    if c not in patch_channels:
                        patch_channels.append(c)
                elif name == "control_change":
                    c, cc, v = event[2:]
                    if not (0 <= c <= 15):
                        continue
                    new_event += [c, cc, v]
                    last_v = control_dict.setdefault((c, cc), 0)
                    if abs(last_v - v) < cc_eps:
                        continue
                    control_dict[(c, cc)] = v
                elif name == "set_tempo":
                    tempo = event[2]
                    if tempo == 0:
                        continue
                    bpm = min(int(self.tempo2bpm(tempo)), 383)
                    new_event += [bpm]
                    if abs(last_bpm - bpm) < tempo_eps:
                        continue
                    last_bpm = bpm
                elif name == "time_signature":
                    nn, dd = event[2:4]
                    if not (1 <= nn <= 16 and 1 <= dd <= 4):
                        continue
                    new_event += [nn - 1, dd - 1]
                elif name == "key_signature":
                    sf, mi = event[2:]
                    if not (-7 <= sf <= 7 and 0 <= mi <= 1):
                        continue
                    new_event += [sf + 7, mi]
                    key_sigs.append(new_event)

                if name in ["note", "time_signature", "key_signature"]:
                    key = tuple(new_event[:-2])
                else:
                    key = tuple(new_event[:-1])

                if c != -1:
                    if c not in channels:
                        channels.append(c)
                    tr_map = track_idx_map[c]
                    if track_idx not in tr_map:
                        tr_map[track_idx] = 0

                if event[0] == "note":
                    cp = tuple(new_event[4:6])
                    if cp in last_notes:
                        last_note_key, last_note = last_notes[cp]
                        last_t = last_note[1] * 16 + last_note[2]
                        last_note[-1] = max(0, min(last_note[-1], t - last_t))
                        if last_note[-1] == 0:
                            event_list.pop(last_note_key)
                    last_notes[cp] = (key, new_event)
                event_list[key] = new_event
        event_list = list(event_list.values())
        empty_channels = [c for c in channels if empty_channels[c]]

        if remap_track_channel:
            patch_channels = []
            channels_count = 0
            channels_map = {9: 9} if 9 in channels else {}
            if remove_empty_channels:
                channels = sorted(channels, key=lambda x: 1 if x in empty_channels else 0)
            for c in channels:
                if c == 9:
                    continue
                channels_map[c] = channels_count
                channels_count += 1
                if channels_count == 9:
                    channels_count = 10
            channels = list(channels_map.values())
            track_count = 0
            track_idx_map_order = [k for k, v in sorted(list(channels_map.items()), key=lambda x: x[1])]
            for c in track_idx_map_order:
                if remove_empty_channels and c in empty_channels:
                    continue
                tr_map = track_idx_map[c]
                for track_idx in tr_map:
                    note_tracks = channel_note_tracks[c]
                    if len(note_tracks) != 0 and track_idx not in note_tracks:
                        continue
                    track_count += 1
                    tr_map[track_idx] = track_count
            for c in track_idx_map_order:
                if not (remove_empty_channels and c in empty_channels):
                    continue
                tr_map = track_idx_map[c]
                for track_idx in tr_map:
                    note_tracks = channel_note_tracks[c]
                    if not (len(note_tracks) != 0 and track_idx not in note_tracks):
                        continue
                    track_count += 1
                    tr_map[track_idx] = track_count

            empty_channels = [channels_map[c] for c in empty_channels]
            track_idx_dict = {}
            key_sigs_new = []
            key_signature_to_add = []
            key_signature_to_remove = []
            for event in event_list:
                name = event[0]
                track_idx = event[3]
                if name == "note":
                    c = event[4]
                    event[4] = channels_map[c]
                    event[3] = track_idx_map[c][track_idx]
                    track_idx_dict.setdefault(event[4], event[3])
                elif name in ["set_tempo", "time_signature"]:
                    event[3] = 0
                elif name == "key_signature":
                    new_channel_track_idxs = []
                    for c, tr_map in track_idx_map.items():
                        if track_idx in tr_map:
                            new_track_idx = tr_map[track_idx]
                            c = channels_map[c]
                            new_channel_track_idx = (c, new_track_idx)
                            if new_track_idx == 0:
                                continue
                            if new_channel_track_idx not in new_channel_track_idxs:
                                new_channel_track_idxs.append(new_channel_track_idx)
                    if len(new_channel_track_idxs) == 0:
                        if event[3] == 0:
                            key_sigs_new.append(event)
                            continue
                        event[3] = -1
                        key_signature_to_remove.append(event)
                        continue
                    c, nt = new_channel_track_idxs[0]
                    event[3] = nt
                    key_sigs_new.append(event)
                    if c == 9:
                        event[4] = 7
                    for c, nt in new_channel_track_idxs[1:]:
                        new_event = [*event]
                        new_event[3] = nt
                        if c == 9:
                            new_event[4] = 7
                        key_sigs_new.append(new_event)
                        key_signature_to_add.append(new_event)
                elif name == "control_change" or name == "patch_change":
                    c = event[4]
                    event[4] = channels_map[c]
                    tr_map = track_idx_map[c]
                    note_tracks = channel_note_tracks[c]
                    if len(note_tracks) != 0 and track_idx not in note_tracks:
                        track_idx = channel_note_tracks[c][0]
                    new_track_idx = tr_map[track_idx]
                    event[3] = new_track_idx
                    if name == "patch_change" and event[4] not in patch_channels:
                        patch_channels.append(event[4])
            for key_sig in key_signature_to_remove:
                event_list.remove(key_sig)
            event_list += key_signature_to_add
            key_sigs = key_sigs_new
            track_to_channels = {}
            for c, tr_map in track_idx_map.items():
                if c not in channels_map:
                    continue
                c = channels_map[c]
                for _, track_idx in tr_map.items():
                    track_to_channels.setdefault(track_idx, [])
                    cs = track_to_channels[track_idx]
                    if c not in cs:
                        cs.append(c)

        if add_default_instr:
            for c in channels:
                if c not in patch_channels and c in track_idx_dict:
                    event_list.append(["patch_change", 0, 0, track_idx_dict[c], c, 0])

        if len(key_sigs) == 0 or all([key_sig[4] == 7 for key_sig in key_sigs]):
            root_key = self.detect_key_signature(note_key_hist)
            if root_key is not None:
                sf = self.key2sf(root_key, 0)
                if len(key_sigs) == 0:
                    for tr, cs in track_to_channels.items():
                        if remap_track_channel and tr == 0:
                            continue
                        new_event = ["key_signature", 0, 0, tr,
                                     (0 if (len(cs) == 1 and cs[0] == 9) else sf) + 7, 0]
                        event_list.append(new_event)
                else:
                    for key_sig in key_sigs:
                        tr = key_sig[3]
                        if tr in track_to_channels:
                            cs = track_to_channels[tr]
                            if len(cs) == 1 and cs[0] == 9:
                                continue
                        key_sig[4] = sf + 7
                        key_sig[5] = 0
            else:
                for key_sig in key_sigs:
                    event_list.remove(key_sig)

        events_name_order = ["time_signature", "key_signature", "set_tempo",
                             "patch_change", "control_change", "note"]
        events_name_order = {name: i for i, name in enumerate(events_name_order)}
        events_order = lambda e: e[1:4] + [events_name_order[e[0]]]
        event_list = sorted(event_list, key=events_order)

        setup_events = {}
        notes_in_setup = False
        for i, event in enumerate(event_list):
            new_event = [*event]
            if event[0] not in ["note", "time_signature"]:
                new_event[1] = 0
                new_event[2] = 0
            has_next = has_pre = False
            if i < len(event_list) - 1:
                next_event = event_list[i + 1]
                has_next = event[1] + event[2] == next_event[1] + next_event[2]
            if notes_in_setup and i > 0:
                pre_event = event_list[i - 1]
                has_pre = event[1] + event[2] == pre_event[1] + pre_event[2]
            if (event[0] == "note" and not has_next) or (notes_in_setup and not has_pre):
                event_list = sorted(setup_events.values(), key=events_order) + event_list[i:]
                break
            else:
                if event[0] == "note":
                    notes_in_setup = True
                if event[0] in ["note", "time_signature", "key_signature"]:
                    key = tuple([event[0]] + event[3:-2])
                else:
                    key = tuple([event[0]] + event[3:-1])
            setup_events[key] = new_event

        last_t1 = 0
        midi_seq = []
        for event in event_list:
            if remove_empty_channels and event[0] in ["control_change", "patch_change"] \
                    and event[4] in empty_channels:
                continue
            cur_t1 = event[1]
            event[1] = event[1] - last_t1
            tokens = self.event2tokens(event)
            if not tokens:
                continue
            midi_seq.append(tokens)
            last_t1 = cur_t1

        if add_bos_eos:
            bos = [self.bos_id] + [self.pad_id] * (self.max_token_seq - 1)
            eos = [self.eos_id] + [self.pad_id] * (self.max_token_seq - 1)
            midi_seq = [bos] + midi_seq + [eos]
        return midi_seq

    def event2tokens(self, event):
        name = event[0]
        params = event[1:]
        if not all([0 <= params[i] < self.event_parameters[p]
                    for i, p in enumerate(self.events[name])]):
            return []
        tokens = [self.event_ids[name]] + [self.parameter_ids[p][params[i]]
                                           for i, p in enumerate(self.events[name])]
        tokens += [self.pad_id] * (self.max_token_seq - len(tokens))
        return tokens

    def tokens2event(self, tokens):
        if tokens[0] not in self.id_events:
            return []
        name = self.id_events[tokens[0]]
        if len(tokens) <= len(self.events[name]):
            return []
        params = tokens[1:]
        params = [params[i] - self.parameter_ids[p][0] for i, p in enumerate(self.events[name])]
        if not all([0 <= params[i] < self.event_parameters[p]
                    for i, p in enumerate(self.events[name])]):
            return []
        return [name] + params

    def detokenize(self, midi_seq):
        ticks_per_beat = 480
        tracks_dict = {}
        t1 = 0
        for tokens in midi_seq:
            if tokens[0] in self.id_events:
                event = self.tokens2event(tokens)
                if not event:
                    continue
                name = event[0]
                t1 += event[1]
                t = t1 * 16 + event[2]
                t = int(t * ticks_per_beat / 16)
                track_idx = event[3]
                event_new = [name, t]
                if name == "note":
                    c, p, v, d = event[4:]
                    d = int(d * ticks_per_beat / 16)
                    event_new += [d, c, p, v]
                elif name in ("control_change", "patch_change"):
                    event_new += event[4:]
                elif name == "set_tempo":
                    event_new += [self.bpm2tempo(event[4])]
                elif name == "time_signature":
                    nn, dd = event[4:]
                    event_new += [nn + 1, dd + 1, 24, 8]
                elif name == "key_signature":
                    sf, mi = event[4:]
                    event_new += [sf - 7, mi]
                else:
                    continue
                if track_idx not in tracks_dict:
                    tracks_dict[track_idx] = []
                tracks_dict[track_idx].append(event_new)
        tracks = [tr for idx, tr in sorted(list(tracks_dict.items()), key=lambda it: it[0])]

        for i in range(len(tracks)):
            track = sorted(tracks[i], key=lambda e: e[1])
            last_note_t = {}
            zero_len_notes = []
            for e in reversed(track):
                if e[0] == "note":
                    t, d, c, p = e[1:5]
                    key = (c, p)
                    if key in last_note_t:
                        d = min(d, max(last_note_t[key] - t, 0))
                    last_note_t[key] = t
                    e[2] = d
                    if d == 0:
                        zero_len_notes.append(e)
            for e in zero_len_notes:
                track.remove(e)
            tracks[i] = track
        return [ticks_per_beat, *tracks]


class MIDITokenizer:
    """Factory: returns MIDITokenizerV1 or MIDITokenizerV2 based on version string."""

    def __new__(cls, version="v2"):
        if version == "v1":
            return MIDITokenizerV1()
        elif version == "v2":
            return MIDITokenizerV2()
        else:
            raise ValueError(f"Unsupported tokenizer version: {version}")
