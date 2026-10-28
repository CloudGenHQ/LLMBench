<p align="center">
  <img src="assets/images/llmbench.png" alt="LLMBench GUI screenshot" width="800">
</p>

# LLMBench

> **Benchmark your local LLM endpoints — fast, pretty, and dependency-free.**

LLMBench is a cross-platform **GUI tool** for measuring the performance of LLM inference servers such as **[Ollama](https://ollama.com)** and OpenAI-compatible servers (**[vLLM](https://github.com/vllm-project/vllm)**, **SGLang**, **llama.cpp**, **LM Studio**, **LocalAI**, **TensorRT-LLM**). Point it at an endpoint, pick a model, hit **Run Benchmark**, and get a full statistical report in seconds.

| | |
|---|---|
| 🐍 **Runtime** | Python 3.8+ |
| 📦 **Dependencies** | *None* — pure standard library |
| 🖥️ **Platforms** | Windows · Linux (any OS with Tkinter) |
| ⚡ **Endpoints** | Ollama (`/api/generate`) · vLLM / SGLang / llama.cpp / LM Studio / LocalAI / TensorRT-LLM (`/v1/chat/completions`) |

---

## ✨ Features

- 🔍 **Auto-discovery** — lists every model served by the endpoint into a dropdown
- 📊 **Six benchmark categories** covering the metrics that actually matter
- 🧮 **Real statistics** — mean, median, min, max, standard deviation, and P95 per metric
- 📈 **Live progress** — watch each test run in real time from the GUI
- 💾 **One-click export** — results as **CSV** or **JSON**
- 🔁 **Settings persistence** — save and reload your endpoint configuration
- 🛡️ **HTTPS support** — with an optional skip-SSL-verification toggle for self-signed certs
- 🚀 **Zero installs** — no `pip install`, no Docker, no config files to write

---

## 🚀 Quick Start

### Windows

Double-click **`run_benchmark_tool_Windows.bat`**, or from a terminal:

```bat
run_benchmark_tool_Windows.bat
```

### Linux

```bash
chmod +x run_benchmark_tool_Linux.sh
./run_benchmark_tool_Linux.sh
```

### Anywhere

You can also run the script directly:

```bash
python3 llm_benchmark.py
```

> The launcher scripts check that a suitable Python interpreter is on your `PATH` (override with the `PYTHON` environment variable) and exit with a clear error if not.

---

## 🖱️ How to Use

1. **Choose the endpoint type** from the dropdown — `Ollama`, `vLLM`, `SGLang`, `llama.cpp`, `LM Studio`, `LocalAI`, or `TensorRT-LLM`. Selecting a type fills in its default URL (e.g. `http://localhost:11434` for Ollama, `http://localhost:8000` for vLLM) and enables the API-key field for OpenAI-compatible servers.
2. **Enter the URL** — scheme, host, and port. A missing `http://` is added for you.
3. **Pick a model** — the dropdown auto-populates from the endpoint's model list.
4. *(Optional)* **Skip SSL verification** for self-signed certificates.
5. **Run Benchmark** — a warm-up request is issued first, then all six tests execute in sequence with live progress.
6. **Review the results table** and **Export CSV / JSON** when done.

---

## 📊 The Benchmark Suite

Every run starts with a **warm-up** request (to prime caches and JIT paths), then executes:

| # | Test | What it measures | Workload |
|---|------|------------------|----------|
| 1 | **TTFT** | Time-to-First-Token | 5 runs · short prompt · 16 max tokens |
| 2 | **Latency** | End-to-end generation time | 5 runs · haiku prompt · 64 max tokens |
| 3 | **Throughput** | Token throughput (tok/s) | 5 runs · long-form prompt · 256 max tokens |
| 4 | **Concurrency** | Wall time per parallelism level | 1 / 2 / 4 / 8 simultaneous requests |
| 5 | **Context Scaling** | TTFT, total latency & throughput (tok/s) vs. context size | 128 / 512 / 2048 / 8192-token prompts |
| 6 | **Consistency** | Determinism of repeated answers | 10 runs · same prompt · unique-response count |

### Statistics per metric

| Statistic | Description |
|-----------|-------------|
| **Mean** | Average across all runs |
| **Median** | Middle value — robust to outliers |
| **Min / Max** | Best and worst observed run |
| **StDev** | Run-to-run variability |
| **P95** | 95th-percentile value |

---

## 💾 Export & Settings

### CSV

```csv
Test,Metric,Unit,Mean,Median,Min,Max,StDev,P95,Raw Values
TTFT,Time to First Token,s,0.1234,0.1200,0.1100,0.1500,0.0123,0.1500,"[0.11, 0.12, 0.12, 0.13, 0.15]"
Throughput,Token Throughput,tok/s,42.1000,41.8000,39.5000,44.2000,1.4500,44.2000,"[...]"
```

### JSON

```json
{
  "endpoint_type": "Ollama",
  "endpoint_url": "http://localhost:11434",
  "model": "llama3.1:8b",
  "timestamp": "2026-10-28T14:32:05.123456",
  "results": [
    {
      "name": "TTFT",
      "metric": "Time to First Token",
      "unit": "s",
      "values": [0.11, 0.12, 0.12, 0.13, 0.15],
      "mean": 0.126,
      "median": 0.12,
      "min": 0.11,
      "max": 0.15,
      "stdev": 0.015,
      "p95": 0.15
    }
  ]
}
```

### Saved settings

**Save Settings** writes `llm_benchmark_settings.json` next to the script (endpoint type, URL, model, SSL flag); **Load Settings** restores it and re-fetches the model list.

---

## 🗂️ Project Structure

```
LLMBench/
├── llm_benchmark.py               # The entire tool — single file, stdlib only
├── run_benchmark_tool_Linux.sh    # Linux launcher
├── run_benchmark_tool_Windows.bat # Windows launcher
├── assets/
│   └── images/
│       ├── llmbench.png           # GUI screenshot (used in this README)
│       └── llmbench.jpg
└── README.md
```

Everything lives in one file: HTTP/streaming helpers, the benchmark engine, and the Tkinter GUI. No packages to build, no virtualenv required.

---

## 🩺 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Model refresh failed` | Make sure the server is running and the URL/port are correct. |
| SSL errors on HTTPS endpoints | Tick **Skip SSL verification** (self-signed certs). |
| `Error: 'python3' not found` (Linux) | Install Python 3.8+, or set `PYTHON=/path/to/python`. |
| GUI doesn't appear (headless Linux) | Tkinter requires an X/Wayland display; run on a desktop session or over X11 forwarding. |
| Timeouts on large models | Each request has a 120 s timeout — very slow first-token times on small hardware may exceed it. |

---

## 📄 License

Free to use, modify, and share. No warranty of any kind.