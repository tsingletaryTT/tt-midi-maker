# From "I wonder if I can make music on my Quietbox?" to shipping a MIDI tool in a weekend

*A practical guide for Tenstorrent developers*

---

You've got a Quietbox 2 running. You've tried the LLM examples, you've compiled a model or two. At some point you look at the thing sitting on your desk and think: *what else can I throw at this?*

This is the story of one answer to that question — and a template for how you can build something similar from first principles using tools every TT developer already has.

---

## The starting idea

The impulse was simple: large language models have learned a lot about music. Music tokens are just sequences. Sequences are what transformers eat. The Quietbox has four P300C chips sitting mostly idle when you're not running something. What happens if you point a MIDI-generation model at all four of them at once and ask it to compose in real time?

That question became `tt-midi-maker` — a multi-track MIDI generation tool with a full MCP server, a streaming loop player, and a coherence pipeline that turns raw model output into something your ears can tolerate.

---

## The building blocks (all open source, all already on your machine)

You don't need to start from scratch. Here's what already exists:

**[skytnt/midi-model](https://github.com/skytnt/midi-model)** — A ~350M parameter LlamaModel trained on MIDI token sequences. It thinks of music as a stream of events: note-on, note-off, tempo, patch changes. The vocabulary maps directly to General MIDI. Apache 2.0 license, checkpoints on HuggingFace.

The model has two parts: `net` is the full 12-layer transformer (~300M params) and `net_token` is a lightweight 3-layer head (~50M params) that runs the per-token sampling step. That split turns out to be the key architectural insight — more on that in a moment.

**[tt-forge-fe](https://github.com/tenstorrent/tt-forge-fe)** — The PyTorch frontend for Tenstorrent hardware. One `forge.compile()` call turns any `nn.Module` into a graph compiled for the P300C mesh. It handles the PCIe dispatch, the multi-chip distribution, and the output buffering. You write normal PyTorch; forge handles the rest.

**[FastMCP](https://github.com/jlowin/fastmcp)** — A decorator-based Python library for building Model Context Protocol servers. One `@mcp.tool()` decorator turns any function into a tool that Claude Desktop (or any MCP client) can call. The whole server is about 100 lines of glue code on top of your actual logic.

**FluidSynth** — A software GM synthesizer that runs as either a one-shot renderer or a persistent ALSA server. `sudo apt install fluidsynth fluid-soundfont-gm` and you have a complete audio pipeline: MIDI in, PCM audio out, speakers.

---

## The architecture decision that made it fast

The naive approach — run the full 12-layer model on hardware for every single token — is slow. Each PCIe dispatch costs about 500ms of latency. At that rate you get ~2 events per second, which means a 16-bar loop takes longer to generate than it takes to play.

The insight is that you don't need hardware every step. The `net_token` head maintains a hidden state from the last hardware call. Between hardware calls, the CPU can keep sampling from that cached state. The tradeoff is musical quality (the context gets staler with each CPU-only step) vs. throughput. At `hw_context_interval=4` — call hardware every 4 events, CPU fills in the 3 between — you get 7.8 events/second while keeping the music coherent. A 16-bar loop at 118 BPM generates in 12 seconds. The loop finishes before the next one needs to start.

This pattern — *use hardware for expensive context updates, CPU for cheap token sampling* — generalizes to any autoregressive model. If you're thinking about what to build next, keep it in mind.

---

## The shape of the code

```
tt_midi_maker/
  generation/
    midi_backend.py      # orchestration: HW vs CPU path, source_midi chaining
    forge_backend.py     # forge.compile wrapper, fixed-shape input (1, 256, 1024)
    skytnt_model.py      # vendored MIDIModel (Apache 2.0)
    hardware.py          # detect_tt_devices() via tt-smi
  coherence/
    scale.py             # quantize notes to key
    harmony.py           # filter against chord tones on beats
    humanize.py          # velocity variation, micro-timing nudge
  server.py              # MCP server: 12 tools, 4 prompts, 4 resources
  stream_player.py       # real-time loop player (FluidSynth + ALSA)
  prompt_engine.py       # LLM → MusicalBlueprint (calls local LLM)
```

The generation backend is the core. Everything else is plumbing. `midi_backend.py` is ~200 lines. The MCP server is ~300 lines of decorators and glue. The coherence passes are ~50 lines each. The whole thing is readable in an afternoon.

---

## What the hardware actually contributes

Without hardware: ~2 ev/s, 35s for a 16-bar loop. Loop ratio 2.5× (takes 2.5× longer than real time).

With 4× P300C at `hw_context_interval=4`: 7.8 ev/s, 12s for the same loop. Loop ratio 0.76× — the machine finishes **before** the current loop ends, so you can queue the next one without a gap.

That 4× speedup is the difference between a tool you can use live and a batch-processing script. The Quietbox isn't doing something your laptop could do given enough time — it's doing something your laptop physically cannot do fast enough for real-time use.

---

## Patterns you can steal

**MCP as the API layer.** Instead of building a REST server or a CLI, expose your tool as an MCP server. Any AI assistant becomes a frontend. The user prompts Claude; Claude calls your tools; your tools call the hardware. No UI to build. The assistant handles intent parsing, argument completion, error explanation, and iteration. `FastMCP` makes a tool registration take three lines.

**Source context chaining.** The model generates better when it can hear what came before. Tokenize the last N bars of the previous output and prepend them to the new prompt. The model reads its own history and continues from it. This is how four separate 8-bar patterns become a coherent 32-bar suite — each generation is seeded by the one before it.

**Post-process the model output.** Raw model output sounds wrong. Notes fall outside the key. Chord tones land on weak beats. Velocities are uniform. Four simple passes — scale quantization, chord-aware filtering, velocity humanization, micro-timing nudge — fix all of this in ~200 lines of music theory. The model handles creativity; the coherence layer handles correctness.

**Split the model at a natural boundary.** The `net` / `net_token` split in skytnt's model isn't the only way to do this, but the principle applies broadly: find the expensive computation (the deep layers that build context) and the cheap computation (the sampling head), and run them at different rates. Hardware for the former; CPU for the latter.

---

## How to start your own version

1. **Pick a model with a clear input/output spec.** Sequence models are easiest — the input and output shapes are fixed, which is what `forge.compile` needs. MIDI, text, audio spectrograms, protein sequences — anything that's a token stream.

2. **Get `forge.compile` working on it first.** Before you build the application, make sure the model compiles and runs on hardware. The compile step is the hard part. Once you have a compiled model that produces correct output, the rest is engineering.

3. **Find the expensive/cheap boundary.** Profile the model. Where does the wall-clock time go? That's where hardware helps most. Everything else runs fine on CPU.

4. **Build the MCP server last.** Get the core generation working as a standalone Python function first. Then wrap it in `@mcp.tool()`. Then test it in Claude Desktop. In that order — don't let the server scaffolding distract from the actual model work.

5. **Demo on real hardware with real output.** Generate something. Render it. Put it on a page. The Smoke & Mirrors suite on the project site took one afternoon to generate and one afternoon to embed. Having a concrete demo changes how people understand what the hardware can do.

---

## The broader point

The Quietbox 2 is a general-purpose AI accelerator sitting on a developer's desk. The models and frameworks to use it are open source. The gap between "I wonder if I could make X" and "I made X" is mostly just willingness to read through a few repos and try things.

Music generation is one answer to "what else can I run on this." The same pipeline — open source model, `forge.compile`, MCP server, local output — applies to image generation, protein folding, code synthesis, scientific simulation, audio processing. Pick the domain you care about. Find the model. Compile it. Ship it.

Nothing leaves the box. That's the point.

---

*`tt-midi-maker` is open source. The code, the generated MIDI files, and this site all live at [github.com/tsingletaryTT/tt-midi-maker](https://github.com/tsingletaryTT/tt-midi-maker).*
