# TunnelBot

Run a 27B LLM (`Qwen3.8-27B` `Q2_K`) with a 32K context window on a free Google Colab Tesla T4 GPU and tunnel it directly to [OpenCode](https://opencode.ai/) on your PC over HTTPS via Cloudflare Quick Tunnels.

```text
Google Colab (Tesla T4, 15 GB VRAM)             Cloudflare Tunnel                  Local PC
┌───────────────────────────────────┐        ┌──────────────────────┐        ┌───────────────────┐
│ Ollama (http://localhost:11434)   │───────▶│ cloudflared          │───────▶│ OpenCode          │
│ Qwen3.8-27B (Q2_K, 32K, q8_0 KV)  │        │ *.trycloudflare.com  │        │ opencode.json     │
└───────────────────────────────────┘        └──────────────────────┘        └───────────────────┘
```

## What the Notebook Does

1. **Verifies the GPU Runtime:** Checks `nvidia-smi` to confirm a Tesla T4 (`15360 MiB` VRAM) is attached.
2. **Installs Dependencies:** Installs `zstd` and [Ollama](https://ollama.com/) inside the Colab VM.
3. **Configures & Starts Ollama:** Kills any stale `ollama` process (`pkill -9 ollama`), sets tuned memory and concurrency environment variables, and starts `ollama serve` in the background.
4. **Pulls & Warms Up the Model:** Runs `hf.co/vcruz305/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-GGUF:Q2_K` with a test prompt (auto-pulling weights on the first run) and checks `ollama ps`.
5. **Creates a Cloudflare Tunnel:** Downloads `cloudflared-linux-amd64`, forwards `http://localhost:11434` with `--http-host-header localhost:11434` (so Ollama accepts tunneled requests), logs to `tunnel.log`, and extracts the active `trycloudflare.com` URL in a separate reusable cell.
6. **Provides Optional Utilities:** Includes commented-out cells to inspect active Ollama environment variables and test OpenAI-compatible function/tool calling locally, plus a stop cell to unload the model from VRAM.

## Prerequisites

- A Google account with access to [Google Colab](https://colab.research.google.com/) (T4 GPU runtime)
- [OpenCode](https://opencode.ai/) installed on your local machine

## Setup Guide

### 1. Run the Notebook in Google Colab

1. Open the TunnelBot notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order:
   - **Connect to T4 runtime:** Confirm `Tesla T4, 15360 MiB` is shown.
   - **Install Ollama:** Installs `zstd` and Ollama.
   - **Start Ollama:** Kills any previous instance, applies the `OLLAMA_*` settings, and starts `ollama serve`.
   - **Test Run:** Downloads/loads the model and runs `ollama ps`. Confirm it shows `100% GPU` and `32768` context.

### 2. Start the Tunnel and Grab the URL

1. Run the `cloudflared` cell to start forwarding port `11434`.
2. Run the URL extraction cell (separated so you can re-run it anytime):
   ```bash
   !grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | tail -1
   ```
3. *(Optional)* Verify connectivity from your PC terminal:
   ```bash
   curl [https://YOUR-URL.trycloudflare.com/v1/models](https://YOUR-URL.trycloudflare.com/v1/models)
   ```

### 3. Register the Provider in OpenCode

Open `~/.config/opencode/opencode.json` (on Windows: `%USERPROFILE%\.config\opencode\opencode.json`) and add the configuration below, replacing `YOUR-URL` with the `trycloudflare.com` URL from Step 2:

```json
{
  "$schema": "[https://opencode.ai/config.json](https://opencode.ai/config.json)",
  "provider": {
    "colab": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Colab Ollama",
      "options": {
        "baseURL": "[https://YOUR-URL.trycloudflare.com/v1](https://YOUR-URL.trycloudflare.com/v1)"
      },
      "models": {
        "hf.co/vcruz305/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-GGUF:Q2_K": {
          "name": "Qwen3.8 27B Q2 (Colab)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "colab/hf.co/vcruz305/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-GGUF:Q2_K"
}
```

> **Multiple providers?** If you already have other providers in `opencode.json`, copy only the `"colab": { ... }` object into your existing `"provider"` section.

## Important: Reasoning Effort Setting

> **Choose `medium` or `low` reasoning effort in OpenCode — do not select `high`.**
>
> OpenCode communicates with Colab using the OpenAI-compatible API format (`@ai-sdk/openai-compatible`). When high reasoning effort is selected, Ollama maps it to `"max"`, which causes a Jinja template error (`Unexpected reasoning effort max`) because Qwen3.8's chat template only accepts `xhigh`, `medium`, and `low`. Press `Ctrl+T` in OpenCode to switch the reasoning effort to **`medium`** or **`low`**.

## Ollama Configuration

| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Sets a 32K context window to fit OpenCode's system prompt, tool schemas, and codebase files. |
| `OLLAMA_FLASH_ATTENTION` | `1` | Enables Flash Attention to reduce memory overhead and speed up inference. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Quantizes the KV cache to 8-bit so the 27B model + 32K context fit 100% in 15 GB T4 VRAM without spilling to CPU RAM. |
| `OLLAMA_NUM_PARALLEL` | `1` | Keeps the entire 32K context dedicated to a single request slot. |
| `OLLAMA_KEEP_ALIVE` | `-1` | Pins the model in VRAM indefinitely to avoid cold-reload delays and tunnel timeouts until manually stopped. |

## Optional Diagnostics & Stopping the Model

- **Verify active Ollama environment variables:** Uncomment and run the `environ` check cell in Colab:
  ```bash
  !tr '\0' '\n' < /proc/$(pgrep -o ollama)/environ | grep OLLAMA
  ```
- **Test tool calling inside Colab:** Uncomment and run the `optional tests` cell to send a sample `get_weather` function-calling request directly to `http://localhost:11434/v1/chat/completions`.
- **Unload the model manually:** Because `OLLAMA_KEEP_ALIVE` is set to `-1`, run the final cell whenever you want to free GPU VRAM:
  ```bash
  !ollama stop hf.co/vcruz305/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-GGUF:Q2_K
  ```
- **New session per tunnel:** Each time you restart the Cloudflare tunnel cell, a new `trycloudflare.com` URL is generated—update `baseURL` in `opencode.json` to match.

## License

This repository is licensed under the [MIT License](LICENSE). Third-party tools and model weights ([Ollama](https://ollama.com/), [cloudflared](https://github.com/cloudflare/cloudflared), and the Hugging Face GGUF model) remain subject to their respective licenses.
