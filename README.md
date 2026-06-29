# Piper Voice Training Guide

> **Train your own text-to-speech voice on any Linux computer — no GPU required.**
>
> This guide covers everything from installing dependencies to generating your first spoken sentence. It works on low-end hardware, CPU-only machines, and is fully accessible to screen-reader users.

---

## Quick Links

| File | What it is |
|------|-----------|
| [`piper-voice-training-guide.md`](piper-voice-training-guide.md) | **The complete guide** — 8 parts, step-by-step, all pitfalls documented |
| [`LICENSE`](LICENSE) | CC0 Public Domain — use freely, modify, share, no attribution required |

---

## Two Ways to Use This Guide

### Option 1: Do It Yourself (Manual)

Open [`piper-voice-training-guide.md`](piper-voice-training-guide.md) and follow it step by step. Every command is spelled out. Every error has a fix.

**The guide is designed for screen-reader users:**
- All text, no images
- Code blocks are plain text (copy-pasteable)
- Use your screen reader's "jump to heading" feature to navigate between sections
- Every warning/pitfall is explicitly called out

**Minimum hardware:** Any x86_64 Linux computer with 4GB RAM and 2GB free disk. No GPU needed.

### Option 2: Use an AI Agent (Assisted)

Have an AI agent do the work for you. Simply share the guide link with an agent and ask it to follow the instructions.

**Step by step for blind/low-vision users:**

1. **Copy this URL:** `https://github.com/jrnwilliams/piper-voice-training-guide`

2. **Choose an agent platform:**

   | Agent | How to use | Cost |
   |-------|-----------|------|
   | **Claude Code** | Install on your computer (terminal-based, screen-reader friendly) | API usage fees |
   | **Hermes Agent** | Open-source, runs locally or connects to Telegram/TUI | Free (bring own API key) |
   | **GitHub Copilot Chat** | In-browser at github.com/copilot | Free tier available |
   | **ChatGPT / Claude.ai** | Paste the guide link into the chat | Free tier available |

3. **Send a message like:**

   > "Please follow the guide at https://github.com/jrnwilliams/piper-voice-training-guide and help me train my own Piper TTS voice on this computer."

4. **The agent will:**
   - Read the guide
   - Run the installation commands
   - Help you prepare your data
   - Start training
   - Troubleshoot any errors

**For Claude Code specifically (recommended for this task):**

```bash
# Install Claude Code (requires Node.js)
npm install -g @anthropic-ai/claude-code

# Navigate to where you want to work
cd ~/my_voice_project

# Start Claude Code
claude

# Then paste:
# "Follow the guide at https://github.com/jrnwilliams/piper-voice-training-guide and help me train a Piper voice"
```

**For Hermes Agent:**

Hermes Agent is an open-source AI assistant by Nous Research. It can connect to Telegram, so you can run training from your phone's screen reader while the agent works on your Linux machine.

- Website: https://hermes-agent.nousresearch.com
- Install: `pip install hermes-agent`
- Run: `hermes setup` then `hermes run`

---

## What You Need Before Starting

- **Audio recordings** of the voice (5-30 minutes, one person speaking)
- **Transcripts** for each clip (what was said)
- **A Linux computer** (any distro — Fedora, Ubuntu, Arch, etc.)
- **No GPU required** — works on CPU, just slower

---

## What You'll End Up With

- An **ONNX model file** (~63 MB) — your trained voice
- A **JSON config file** — goes alongside the ONNX
- **Sample WAV files** — hear your voice saying test phrases

Your voice can be used with:
- VoiceOver on iOS/macOS
- Any Piper-compatible TTS engine
- Home Assistant / Rhasspy
- The `piper` command-line tool

---

## Training Time Estimates (CPU-Only)

| Phase | Time |
|-------|------|
| Setup | 10-30 minutes |
| 10 epochs | ~35 minutes |
| 50 epochs | ~3 hours |
| 100 epochs | ~6 hours |
| 500 epochs | ~2-3 days |

You can stop at any point and still have a usable voice.

---

## Credits

This guide was developed through extensive real-world troubleshooting on a CPU-only Linux system (Fedora 44, no GPU). Every fix was discovered empirically because the official [Piper repository](https://github.com/rhasspy/piper) was archived in October 2025 and all official documentation is now broken.

Created with help from **Hermes Agent** (Nous Research).

**License:** CC0 Public Domain — use it however you want.
