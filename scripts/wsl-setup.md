# WSL2 setup for a 16GB / 4GB-VRAM machine

## 1. Cap WSL memory (Windows side)

Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=10GB
processors=6
swap=8GB
```

Then in PowerShell: `wsl --shutdown` and reopen your terminal.
Without this, WSL2 takes half your RAM by default and Docker fights the browser.

## 2. Verify CUDA reaches WSL

```bash
nvidia-smi                 # should list the RTX and 4096MiB
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If `nvidia-smi` fails, update the NVIDIA driver on Windows (not inside WSL —
never install a driver inside WSL).

## 3. Ollama

Install on the Windows side, not inside WSL — it gets direct GPU access and
WSL reaches it over `http://localhost:11434` anyway.

```powershell
ollama pull qwen2.5-coder:3b-instruct-q4_K_M
ollama run qwen2.5-coder:3b-instruct-q4_K_M "explain a null pointer exception in one line"
```

Watch `nvidia-smi` while it answers. If VRAM use pins at 4096MiB and tokens
crawl, drop `num_ctx` to 4096 in `.env`.

## 4. Keep the repo on the Linux filesystem

Work in `~/projects/codemind-ai`, **not** `/mnt/c/...`. Cross-filesystem I/O in
WSL2 is roughly 10x slower and tree-sitter parsing a 10k-file repo will crawl.
