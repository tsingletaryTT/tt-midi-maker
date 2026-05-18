# Installation Guide — tt-midi-maker

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 recommended |
| pip | 23+ | `pip install --upgrade pip` |
| A Tenstorrent device | optional | N150 / N300 / T3000; CPU fallback is used without one |
| tt-smi | optional | Needed for hardware detection (`tt-smi -s`) |
| tt-forge | optional | Needed for on-device model compilation |
| An LLM endpoint | required | Any OpenAI-compatible `/chat/completions` server |

The server starts and generates MIDI without TT hardware using a CPU fallback path.
An LLM endpoint is required for blueprint generation and MIDI analysis — see
"LLM setup" below.

---

## 1. Clone / obtain the source

```bash
git clone https://github.com/tenstorrent/tt-midi-maker
cd tt-midi-maker
```

---

## 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
```

---

## 3. Install the package

**Development install** (editable, includes test dependencies):

```bash
pip install -e ".[dev]"
```

**Production install** (no test tools):

```bash
pip install -e .
```

This installs all runtime dependencies declared in `pyproject.toml`:

| Package | Purpose |
|---|---|
| `mcp>=1.27.0` | MCP server framework (FastMCP) |
| `mido>=1.3.0` | MIDI file read/write |
| `miditok>=3.0.0` | MIDI tokenisation (REMI+) |
| `transformers>=4.40.0` | Aria model inference |
| `torch>=2.0.0` | Tensor backend for Aria |
| `pydantic>=2.0.0` | Blueprint and config validation |
| `httpx>=0.27.0` | Async HTTP client for LLM calls |
| `pyyaml>=6.0` | `config/*.yaml` loading |

---

## 4. LLM setup

tt-midi-maker needs an OpenAI-compatible LLM to turn prompts into musical blueprints
and to answer analytical questions about MIDI files.

### Option A — Local LLM via Ollama (recommended for TT hardware users)

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3          # or llama3.1, mistral, etc.
ollama serve               # starts on :11434

export MIDI_LLM_URL=http://localhost:11434/v1
export MIDI_LLM_MODEL=qwen3
```

### Option B — Local LLM via vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port 8000

export MIDI_LLM_URL=http://localhost:8000/v1
export MIDI_LLM_MODEL=Qwen/Qwen3-8B
```

### Option C — Remote API (OpenAI, Together, Groq, etc.)

```bash
export MIDI_LLM_URL=https://api.openai.com/v1
export MIDI_LLM_MODEL=gpt-4o-mini
# Set your provider's API key as needed:
export OPENAI_API_KEY=sk-...
```

### Making env vars permanent

Add the exports to `~/.bashrc` or `~/.zshrc`, or create a `.env` file and source it
before starting the server.

---

## 5. Tenstorrent hardware setup (optional)

If you have a Tenstorrent device, install the tt-forge runtime to accelerate generation.

### Check device is visible

```bash
tt-smi -s          # should list your device with status "available"
```

### Install tt-forge

Follow the official guide at https://docs.tenstorrent.com/tt-forge — the package is
not on PyPI and must be installed from the Tenstorrent release artifacts.

Once installed, the server will detect your device via `tt-smi -s` at startup and
attempt to compile the Aria model with `forge.compile()`. If compilation fails or
no device is found, generation falls back to CPU automatically.

### Hugepages (required for tt-metal / tt-forge)

```bash
# Check current setting
cat /proc/sys/vm/nr_hugepages

# Set 2 GB of hugepages (example for a single N300)
sudo sh -c 'echo 512 > /proc/sys/vm/nr_hugepages'

# Make permanent
echo 'vm.nr_hugepages = 512' | sudo tee /etc/sysctl.d/99-hugepages.conf
sudo sysctl --system
```

---

## 6. Verify installation

```bash
# Run the test suite (all 98 tests should pass)
pytest tests/ -v

# Start the server and confirm it binds
python -m tt_midi_maker &
sleep 2
curl -s http://127.0.0.1:8000/mcp   # should return MCP metadata JSON
kill %1
```

Expected test output:
```
========================= 98 passed, 1 warning in ~1.5s =========================
```

The single warning is a harmless miditok configuration notice about attribute controls.

---

## 7. Start the server

```bash
# Foreground (shows logs)
python -m tt_midi_maker

# Background
python -m tt_midi_maker &

# Using the installed entry point
tt-midi-maker
```

The server listens on `http://127.0.0.1:8000` using the **streamable-HTTP** MCP
transport. Output MIDI files go to `~/Music/tt-midi-maker/` (created automatically).

---

## 8. Connect an MCP client

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tt-midi-maker": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Restart Claude Desktop. The `generate_midi`, `describe_midi`, and other tools will
appear in Claude's tool list.

### Claude Code (CLI)

```bash
claude mcp add tt-midi-maker --transport http --url http://127.0.0.1:8000/mcp
```

### Cursor / Windsurf / other editors

Use the HTTP transport URL `http://127.0.0.1:8000/mcp`. Consult your editor's MCP
documentation for the exact config format.

---

## 9. First generation

With the server running and Claude Desktop connected:

1. Open a new Claude conversation
2. Ask: *"Generate a calm 8-bar lo-fi hip hop loop in C minor"*
3. Claude calls `set_musical_context` then `generate_midi`
4. The `.mid` file path is returned — open it in your DAW

Or call the tool directly:

```python
# Example using the MCP Python SDK
import mcp

async with mcp.ClientSession("http://127.0.0.1:8000/mcp") as session:
    result = await session.call_tool("generate_midi", {
        "prompt": "calm lo-fi hip hop, C minor, dusty drums and sparse piano",
        "mode": "loop",
        "bars": 8,
    })
    print(result)  # {"file_path": "~/Music/tt-midi-maker/1716000000.mid", ...}
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tt_midi_maker'`

Activate your virtual environment and confirm the package is installed:

```bash
source .venv/bin/activate
pip show tt-midi-maker
```

If not installed, run `pip install -e .` from the project root.

---

### `httpx.ConnectError` when generating

The LLM endpoint is not reachable. Check:

```bash
curl $MIDI_LLM_URL/models     # should return a model list
```

Ensure your LLM server is running and `MIDI_LLM_URL` is set correctly.

---

### `[CONTEXT_NOT_SET]` error from generate_midi

The LLM returned something that could not be parsed as a MusicalBlueprint. This usually
means the LLM refused or gave a malformed JSON response.

- Try a more specific prompt
- Check the LLM is a capable instruction-following model (7B+ recommended)
- Set context explicitly first with `set_musical_context`

---

### No TT device detected / slow generation

Check hardware visibility:

```bash
tt-smi -s        # lists detected devices and their status
```

If no devices appear, generation falls back to CPU (slower but functional).
If devices appear but generation still uses CPU, check tt-forge is installed:

```bash
python -c "import forge; print(forge.__version__)"
```

---

### `UserWarning: Attribute controls are not compatible...` from miditok

This is a harmless warning from miditok's REMI tokenizer configuration. It does not
affect output quality. It will be resolved when miditok updates its API.

---

### Server port conflict

If port 8000 is already in use, set a different port by starting uvicorn manually:

```bash
python -c "
from tt_midi_maker.server import mcp
mcp.run(transport='streamable-http', host='127.0.0.1', port=8001)
"
```

Then update your MCP client URL to `:8001/mcp`.

---

### Stale `/dev/shm/tenstorrent*` files blocking device init

After a crashed workload, stale shared-memory files can prevent device initialisation:

```bash
ls /dev/shm/tenstorrent*        # check for stale files
rm /dev/shm/tenstorrent*        # remove them
tt-smi -r                        # reset devices
```

---

## Uninstall

```bash
pip uninstall tt-midi-maker
```

Output files in `~/Music/tt-midi-maker/` are not removed automatically.
