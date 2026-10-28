#!/usr/bin/env python3
"""
LLM Endpoint Benchmark Tool
============================

A cross-platform (Windows / Linux) GUI tool for benchmarking LLM inference
endpoints such as **Ollama** and OpenAI-compatible servers (**vLLM**,
**SGLang**, **llama.cpp**, **LM Studio**, **LocalAI**, **TensorRT-LLM**).

Features
--------
* Explicit Connect button with live status (green / red)
* Auto-discover models from the endpoint (on connect or refresh)
* Six benchmark categories:
    1. Time-To-First-Token (TTFT)
    2. Full-generation latency
    3. Throughput (tokens / second)
    4. Concurrency  (4 / 8 / 16 parallel requests)
    5. Context-length scaling  (100 → 2 000 chars)
    6. Consistency / variance  (10 repeated runs)
* Live results table in the GUI
* One-click export to CSV or JSON
* Persistent run log – every run is appended to ``logs/llm_bench_runs.log``
  with all chosen settings, and a per-run HTML report is saved to
  ``logs/reports/`` (both are kept across app launches)

Usage
-----
    python llm_benchmark.py

Requirements
------------
Python 3.8+  (standard library only – no pip installs needed)
"""

from __future__ import annotations

import csv
import html
import http.client
import json
import os
import queue
import statistics
import threading
import time
import tkinter as tk
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

# ══════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════

__version__ = "1.0.0"  # application version (shown in the GUI title, reports, and logs)

TIMEOUT = 120  # seconds per HTTP request
CONNECT_TIMEOUT = 10  # seconds for connection checks / model discovery
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "llm_benchmark_settings.json")

# -- Test 1: TTFT --
TTFT_PROMPT = "Say hello."
TTFT_MAX_TOKENS = 10
TTFT_RUNS = 5

# -- Test 2: Latency --
LATENCY_PROMPT = "Write a short paragraph about the history of computing."
LATENCY_MAX_TOKENS = 100
LATENCY_RUNS = 5

# -- Test 3: Throughput --
THROUGHPUT_PROMPT = "Explain the theory of relativity in detail."
THROUGHPUT_MAX_TOKENS = 200
THROUGHPUT_RUNS = 5

# -- Test 4: Concurrency --
CONCURRENCY_LEVELS = [4, 8, 16]
CONCURRENCY_PROMPT = "Tell me a short story about a robot."
CONCURRENCY_MAX_TOKENS = 100

# -- Test 5: Context scaling --
CONTEXT_LENGTHS = [100, 500, 1000, 2000]
CONTEXT_MAX_TOKENS = 50
CONTEXT_RUNS = 3

# Context window sizes offered in the GUI (display label, token count)
CONTEXT_SIZES = [
    ("1k", 1024),
    ("2k", 2048),
    ("4k", 4096),
    ("8k", 8192),
    ("16k", 16384),
    ("32k", 32768),
    ("64k", 65536),
    ("128k", 131072),
]
DEFAULT_CONTEXT_SIZE = 8192

# -- Test 6: Consistency --
CONSISTENCY_PROMPT = "What is the capital of France?"
CONSISTENCY_MAX_TOKENS = 20
CONSISTENCY_RUNS = 10

# -- Endpoint types offered in the GUI (display label, protocol family) --
#   "openai" -> OpenAI-compatible HTTP API  (/v1/chat/completions, /v1/models)
#   "ollama" -> Ollama native API           (/api/generate, /api/tags)
#
# Most modern local servers (vLLM, SGLang, llama.cpp, LM Studio, LocalAI,
# TensorRT-LLM) speak the OpenAI protocol and therefore share one code path;
# each is listed explicitly so the user picks the exact server they run.
ENDPOINT_TYPES: List[Tuple[str, str]] = [
    ("Ollama", "ollama"),
    ("vLLM", "openai"),
    ("SGLang", "openai"),
    ("llama.cpp", "openai"),
    ("LM Studio", "openai"),
    ("LocalAI", "openai"),
    ("TensorRT-LLM", "openai"),
]

# Sensible default base URL per endpoint type (applied when the type changes).
ENDPOINT_DEFAULT_URLS: Dict[str, str] = {
    "Ollama": "http://localhost:11434",
    "vLLM": "http://localhost:8000",
    "SGLang": "http://localhost:30000",
    "llama.cpp": "http://localhost:8080",
    "LM Studio": "http://localhost:1234",
    "LocalAI": "http://localhost:8080",
    "TensorRT-LLM": "http://localhost:8000",
}

# Older settings files stored the protocol name ("vllm" / "ollama") instead of
# a specific type label. Map those to a concrete label when loading.
_LEGACY_TYPE_MAP: Dict[str, str] = {"vllm": "vLLM", "ollama": "Ollama"}


def endpoint_protocol(label: str) -> str:
    """Return the protocol family (``"openai"`` | ``"ollama"``) for a type label.

    Legacy protocol names are resolved correctly; anything unrecognised falls
    back to the OpenAI-compatible protocol (the most common case).
    """
    label = _LEGACY_TYPE_MAP.get(label, label)
    for name, proto in ENDPOINT_TYPES:
        if name == label:
            return proto
    return "openai"


# ══════════════════════════════════════════════
#  DATA  CLASSES
# ══════════════════════════════════════════════

@dataclass
class TestResult:
    """One row in the results table."""
    name: str
    metric: str
    unit: str
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    stdev: float = 0.0
    p95: float = 0.0

    def compute_stats(self) -> None:
        if not self.values:
            return
        self.mean = statistics.mean(self.values)
        self.median = statistics.median(self.values)
        self.min = min(self.values)
        self.max = max(self.values)
        self.stdev = statistics.stdev(self.values) if len(self.values) > 1 else 0.0
        s = sorted(self.values)
        self.p95 = s[min(int(0.95 * (len(s) - 1)), len(s) - 1)]


@dataclass
class BenchmarkReport:
    """Full report – serialisable to dict / CSV / JSON."""
    endpoint_type: str
    endpoint_url: str
    model: str
    timestamp: str
    context_size: int = 8192
    results: List[TestResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_type": self.endpoint_type,
            "endpoint_url": self.endpoint_url,
            "model": self.model,
            "timestamp": self.timestamp,
            "context_size": self.context_size,
            "tool_version": __version__,
            "results": [asdict(r) for r in self.results],
        }

    def to_csv(self, filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Test", "Metric", "Unit", "Mean", "Median",
                        "Min", "Max", "StDev", "P95", "Raw Values"])
            for r in self.results:
                w.writerow([
                    r.name, r.metric, r.unit,
                    f"{r.mean:.4f}", f"{r.median:.4f}",
                    f"{r.min:.4f}", f"{r.max:.4f}",
                    f"{r.stdev:.4f}", f"{r.p95:.4f}",
                    json.dumps(r.values),
                ])


# ══════════════════════════════════════════════
#  RUN  LOG
# ══════════════════════════════════════════════
#
#  All run artifacts live in a single, well-structured ``logs`` folder:
#
#      logs/
#      ├── llm_bench_runs.log          # text log – one entry per run
#      └── reports/
#          └── llm_bench_report_*.html # one self-contained HTML per run
#
#  The folder is created next to the app, or under the user's home
#  directory if that location is not writable.  It is kept across app
#  launches, so the user can find all of their runs in one place.

def _resolve_logs_dir() -> str:
    """Find a writable logs directory (next to the app, else a home fallback).

    Creates the directory (and its ``reports`` subfolder) if needed and
    returns the first candidate that is actually writable.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(app_dir, "logs"),
        os.path.join(os.path.expanduser("~"), "llm_bench", "logs"),
    ]
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            # Verify it is writable by creating and removing a probe file.
            probe = os.path.join(path, ".write_test")
            with open(probe, "w", encoding="utf-8"):
                pass
            os.remove(probe)
            return path
        except OSError:
            continue
    return candidates[-1]


def _migrate_legacy_artifacts() -> None:
    """One-time migration: fold legacy app-dir artifacts into the logs/ folder.

    Older versions wrote the run log and per-run HTML reports directly next
    to the app.  If any of those files still exist, move them into the
    well-structured logs folder so all run history lives in one place.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) Legacy run log -> logs/llm_bench_runs.log
    legacy_log = os.path.join(app_dir, "llm_bench_runs.log")
    if os.path.isfile(legacy_log):
        try:
            if os.path.exists(LOG_FILE):
                with open(legacy_log, "r", encoding="utf-8") as src, \
                        open(LOG_FILE, "a", encoding="utf-8") as dst:
                    dst.write(src.read())
                os.remove(legacy_log)
            else:
                os.replace(legacy_log, LOG_FILE)
        except OSError:
            pass  # best effort – leave the legacy file in place

    # 2) Legacy HTML reports -> logs/reports/
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
    except OSError:
        return
    for name in os.listdir(app_dir):
        if not (name.startswith("llm_bench_report_")
                and name.endswith(".html")):
            continue
        src = os.path.join(app_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(REPORTS_DIR, name)
        counter = 1
        while os.path.exists(dst):
            dst = os.path.join(REPORTS_DIR, f"{name[:-5]}_{counter}.html")
            counter += 1
        try:
            os.replace(src, dst)
        except OSError:
            pass  # best effort – leave the legacy file in place


LOGS_DIR = _resolve_logs_dir()
LOG_FILE = os.path.join(LOGS_DIR, "llm_bench_runs.log")
REPORTS_DIR = os.path.join(LOGS_DIR, "reports")
_migrate_legacy_artifacts()


def _mask_key(key: str) -> str:
    """Mask an API key so it is safe to write to the log."""
    if not key:
        return "(none)"
    return key[:4] + "*" * max(len(key) - 4, 3)


def _log_session_start() -> None:
    """Mark the start of an app session in the log."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n--- session started "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(LLMBench v{__version__}) ---\n")
    except OSError:
        pass


def _write_log(lines: List[str]) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as e:
        print(f"Could not write run log {LOG_FILE}: {e}")


def _log_run(report: BenchmarkReport, settings: Dict[str, str]) -> Optional[str]:
    """Append one completed benchmark run (settings + results) to the log.

    Also writes a self-contained HTML report for the run and returns its
    file path (or ``None`` if the HTML could not be written).
    """
    lines = ["=" * 78,
             f"RUN {report.timestamp}",
             "-" * 78]
    lines += [f"  {key:<16}: {value}" for key, value in settings.items()]
    lines.append("-" * 78)
    lines.append(f"  {'Test':<30} {'Metric':<26} {'Unit':<6} {'Mean':>8} "
                 f"{'Median':>8} {'Min':>8} {'Max':>8} {'StDev':>8} {'P95':>8}")
    for r in report.results:
        lines.append(f"  {r.name[:30]:<30} {r.metric[:26]:<26} {r.unit:<6} "
                     f"{r.mean:8.4f} {r.median:8.4f} {r.min:8.4f} "
                     f"{r.max:8.4f} {r.stdev:8.4f} {r.p95:8.4f}")
    lines += ["=" * 78, ""]
    _write_log(lines)
    return _log_run_html(report, settings)


def _log_run_failed(settings: Dict[str, str], error: str) -> None:
    """Append a failed benchmark run to the log."""
    lines = ["=" * 78,
             f"RUN {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  (FAILED)",
             "-" * 78]
    lines += [f"  {key:<16}: {value}" for key, value in settings.items()]
    lines.append("-" * 78)
    lines.append(f"  Error: {error}")
    lines += ["=" * 78, ""]
    _write_log(lines)


# ══════════════════════════════════════════════
#  HTML  REPORT
# ══════════════════════════════════════════════

#: Inline CSS for the generated HTML report. Kept as a plain (non-f) string
#: so the ``{`` / ``}`` braces do not need escaping.
_HTML_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #f4f6fa;
  color: #2b3242;
  line-height: 1.5;
}
.wrap { max-width: 1000px; margin: 0 auto; padding: 28px 20px 60px; }
header.hero {
  background: linear-gradient(135deg, #3b5bdb 0%, #4f8ef7 100%);
  color: #fff;
  border-radius: 14px;
  padding: 26px 28px;
  box-shadow: 0 8px 24px rgba(59, 91, 219, .25);
}
header.hero h1 { margin: 0 0 6px; font-size: 24px; font-weight: 700; }
header.hero .sub { opacity: .92; font-size: 15px; }
header.hero .meta { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 13px; }
header.hero .meta b { font-weight: 600; opacity: .85; }
h2 { font-size: 18px; margin: 34px 0 14px; color: #2b3242; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }
.card {
  background: #fff; border-radius: 12px; padding: 16px 18px;
  box-shadow: 0 2px 8px rgba(20, 30, 60, .06);
}
.card-title { font-size: 12px; color: #8a93a3; text-transform: uppercase; letter-spacing: .04em; }
.card-value { font-size: 28px; font-weight: 700; margin-top: 6px; color: #2b3242; }
.card-unit { font-size: 12px; color: #8a93a3; margin-top: 2px; }
.charts { display: grid; grid-template-columns: 1fr; gap: 16px; }
.chart-card {
  background: #fff; border-radius: 12px; padding: 18px 20px;
  box-shadow: 0 2px 8px rgba(20, 30, 60, .06);
}
.chart-card h3 { margin: 0 0 10px; font-size: 15px; color: #2b3242; }
.chart-unit { color: #8a93a3; font-weight: 400; font-size: 13px; }
table {
  width: 100%; border-collapse: collapse; background: #fff;
  border-radius: 12px; overflow: hidden;
  box-shadow: 0 2px 8px rgba(20, 30, 60, .06);
}
th, td { padding: 10px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid #eef1f6; }
thead th {
  background: #f0f3f9; color: #5a6478; font-weight: 600;
  text-transform: uppercase; font-size: 11px; letter-spacing: .04em;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #f7f9fc; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.t { font-weight: 500; }
.settings td:first-child { width: 180px; color: #8a93a3; }
footer { margin-top: 36px; text-align: center; color: #a0a8b8; font-size: 12px; }
"""


def _svg_bar_chart(items: List[Tuple[str, float]], color: str,
                   width: int = 760, height: int = 260) -> str:
    """Render a dependency-free SVG bar chart from ``(label, value)`` pairs."""
    if not items:
        return ""
    left, right, top, bottom = 52, 16, 28, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(v for _, v in items) or 1.0
    n = len(items)
    slot = plot_w / n
    bar_w = slot * 0.6

    parts = [f'<svg viewBox="0 0 {width} {height}" '
             f'xmlns="http://www.w3.org/2000/svg" '
             f'role="img" style="width:100%;height:auto;">']
    # horizontal gridlines + y-axis value labels
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#e6e9ef" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#8a93a3">{max_val * frac:.0f}</text>')
    # bars, value labels and (rotated) x-axis labels
    for i, (label, value) in enumerate(items):
        x = left + i * slot + (slot - bar_w) / 2
        bar_h = (value / max_val) * plot_h
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" rx="3" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" '
            f'text-anchor="middle" font-size="11" font-weight="600" '
            f'fill="#3a4256">{value:.1f}</text>')
        cx = x + bar_w / 2
        cy = top + plot_h + 14
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="end" '
            f'font-size="11" fill="#5a6478" '
            f'transform="rotate(-32 {cx:.1f} {cy:.1f})">{html.escape(label)}</text>')
    # baseline
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" '
        f'y2="{top + plot_h}" stroke="#c9cfda" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def _generate_html_report(report: BenchmarkReport,
                          settings: Dict[str, str]) -> str:
    """Build a self-contained HTML report (inline CSS + SVG charts)."""
    esc = html.escape

    # group results by unit so each chart stays on a single scale
    groups: Dict[str, List[TestResult]] = {}
    for r in report.results:
        groups.setdefault(r.unit, []).append(r)

    def _mean_of(unit: str) -> Optional[float]:
        vals = [r.mean for r in groups.get(unit, [])]
        return statistics.mean(vals) if vals else None

    def _find(sub: str) -> Optional[TestResult]:
        for r in report.results:
            if sub.lower() in r.metric.lower():
                return r
        return None

    # ── summary cards ─────────────────────────────────────────────
    def _card(title: str, value: Optional[float], unit: str, accent: str) -> str:
        if value is None:
            return ""
        return (f'<div class="card" style="border-top:3px solid {accent};">'
                f'<div class="card-title">{esc(title)}</div>'
                f'<div class="card-value">{value:.2f}</div>'
                f'<div class="card-unit">{esc(unit)}</div></div>')

    ttft = _find("TTFT")
    tpot = _find("TPOT")
    e2e = _find("E2E")
    cards_html = "".join(c for c in (
        _card("Throughput", _mean_of("tok/s"), "tokens/sec", "#4f8ef7"),
        _card("Time to First Token", ttft.mean if ttft else None, "ms", "#f7a14f"),
        _card("Time per Output Token", tpot.mean if tpot else None, "ms", "#f7a14f"),
        _card("E2E Latency", e2e.mean if e2e else None, "ms", "#f7a14f"),
        _card("Consistency", _mean_of("%"), "%", "#5fd08a"),
    ) if c)

    # ── charts ────────────────────────────────────────────────────
    chart_defs = [
        ("tok/s", "Throughput by test", "tokens/sec", "#4f8ef7"),
        ("ms", "Latency by test", "ms", "#f7a14f"),
        ("%", "Consistency by test", "%", "#5fd08a"),
    ]
    charts_html = ""
    for unit, title, u, color in chart_defs:
        if unit in groups:
            items = [(r.name, r.mean) for r in groups[unit]]
            charts_html += (
                f'<div class="chart-card">'
                f'<h3>{esc(title)} <span class="chart-unit">({esc(u)})</span></h3>'
                f'{_svg_bar_chart(items, color)}</div>')

    # ── settings table ────────────────────────────────────────────
    settings_rows = "".join(
        f'<tr><td class="t">{esc(k)}</td><td>{esc(str(v))}</td></tr>'
        for k, v in settings.items())

    # ── full results table ────────────────────────────────────────
    rows = ""
    for r in report.results:
        rows += (
            f'<tr><td class="t">{esc(r.name)}</td>'
            f'<td class="t">{esc(r.metric)}</td>'
            f'<td>{esc(r.unit)}</td>'
            f'<td class="num">{r.mean:.3f}</td>'
            f'<td class="num">{r.median:.3f}</td>'
            f'<td class="num">{r.min:.3f}</td>'
            f'<td class="num">{r.max:.3f}</td>'
            f'<td class="num">{r.stdev:.3f}</td>'
            f'<td class="num">{r.p95:.3f}</td></tr>')

    body = f"""
<header class="hero">
  <h1>LLM Benchmark Report</h1>
  <div class="sub">{esc(report.model)}</div>
  <div class="meta">
    <span><b>Endpoint:</b> {esc(report.endpoint_type)} &mdash; {esc(report.endpoint_url)}</span>
    <span><b>Context:</b> {report.context_size} tokens</span>
    <span><b>Run:</b> {esc(report.timestamp)}</span>
    <span><b>Tool:</b> LLMBench v{__version__}</span>
  </div>
</header>

<h2>Summary</h2>
<div class="cards">{cards_html}</div>

<h2>Charts</h2>
<div class="charts">{charts_html}</div>

<h2>Settings</h2>
<table class="settings"><tbody>{settings_rows}</tbody></table>

<h2>All Results</h2>
<table>
  <thead><tr>
    <th>Test</th><th>Metric</th><th>Unit</th>
    <th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>StDev</th><th>P95</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>

<footer>Generated by LLMBench v{__version__} on {esc(report.timestamp)}</footer>
"""

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc('LLM Benchmark — ' + report.model)}</title>\n"
        f"<style>{_HTML_CSS}</style>\n"
        "</head>\n"
        '<body>\n'
        f'<div class="wrap">{body}</div>\n'
        "</body>\n"
        "</html>"
    )


def _log_run_html(report: BenchmarkReport,
                  settings: Dict[str, str]) -> Optional[str]:
    """Write a self-contained HTML report for one run; return its path.

    Reports are stored in ``logs/reports/``.  The filename is timestamped
    and, if that exact name already exists (two runs in the same second),
    a numeric suffix is appended so no report is ever overwritten.
    """
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"llm_bench_report_{stamp}.html")
        counter = 1
        while os.path.exists(path):
            path = os.path.join(REPORTS_DIR,
                                f"llm_bench_report_{stamp}_{counter}.html")
            counter += 1
        with open(path, "w", encoding="utf-8") as f:
            f.write(_generate_html_report(report, settings))
        return path
    except OSError as e:
        print(f"Could not write HTML report: {e}")
        return None


# ══════════════════════════════════════════════
#  HTTP  UTILITIES
# ══════════════════════════════════════════════

def parse_url(url: str) -> Tuple[str, int, bool]:
    """Return ``(host, port, is_https)`` from a user-supplied URL."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    p = urlparse(url)
    port = p.port or (443 if p.scheme == "https" else 80)
    return p.hostname or "localhost", port, p.scheme == "https"


def _conn(host: str, port: int, is_https: bool,
          skip_ssl: bool = False,
          timeout: float = TIMEOUT) -> http.client.HTTPConnection:
    if is_https:
        import ssl
        ctx = ssl._create_unverified_context() if skip_ssl else None
        return http.client.HTTPSConnection(host, port,
                                           timeout=timeout, context=ctx)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def http_get_json(host: str, port: int, path: str,
                  is_https: bool = False,
                  skip_ssl: bool = False,
                  headers: Optional[Dict[str, str]] = None,
                  timeout: float = TIMEOUT) -> Any:
    """GET *path* and return the parsed JSON body."""
    c = _conn(host, port, is_https, skip_ssl, timeout)
    try:
        c.request("GET", path, headers=headers or {})
        resp = c.getresponse()
        body = resp.read().decode("utf-8")
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
        return json.loads(body)
    finally:
        c.close()


# ══════════════════════════════════════════════
#  STREAMING  HELPERS
# ══════════════════════════════════════════════
#
#  Both generators yield  (token_text, is_first_token, metadata_dict)
#  so the benchmark engine can measure TTFT and token counts uniformly.

def _stream_vllm(
    host: str, port: int, model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    is_https: bool = False,
    skip_ssl: bool = False,
    api_key: str = "",
) -> Generator[Tuple[str, bool, Dict], None, None]:
    """Stream from a vLLM / OpenAI-compatible ``/v1/chat/completions``.

    If *api_key* is non-empty an ``Authorization: Bearer <key>`` header is
    sent; otherwise the request is made without authentication.
    """
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    })
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    c = _conn(host, port, is_https, skip_ssl)
    try:
        c.request("POST", "/v1/chat/completions", body=body,
                  headers=headers)
        resp = c.getresponse()
        if resp.status != 200:
            raise RuntimeError(
                f"vLLM {resp.status}: {resp.read().decode()[:500]}")

        first = True
        meta: Dict[str, Any] = {}
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "usage" in chunk:
                meta["usage"] = chunk["usage"]
            for ch in chunk.get("choices", []):
                text = ch.get("delta", {}).get("content", "")
                if text:
                    yield text, first, meta
                    first = False
    finally:
        c.close()


def _stream_ollama(
    host: str, port: int, model: str,
    prompt: str,
    max_tokens: int,
    is_https: bool = False,
    skip_ssl: bool = False,
) -> Generator[Tuple[str, bool, Dict], None, None]:
    """Stream from an Ollama ``/api/generate`` endpoint."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens},
    })
    c = _conn(host, port, is_https, skip_ssl)
    try:
        c.request("POST", "/api/generate", body=body,
                  headers={"Content-Type": "application/json"})
        resp = c.getresponse()
        if resp.status != 200:
            raise RuntimeError(
                f"Ollama {resp.status}: {resp.read().decode()[:500]}")

        first = True
        meta: Dict[str, Any] = {}
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("done"):
                meta["eval_count"] = chunk.get("eval_count", 0)
                meta["eval_duration"] = chunk.get("eval_duration", 0)
                meta["prompt_eval_count"] = chunk.get("prompt_eval_count", 0)
                break
            text = chunk.get("response", "")
            if text:
                yield text, first, meta
                first = False
    finally:
        c.close()


# ══════════════════════════════════════════════
#  MODEL  DISCOVERY
# ══════════════════════════════════════════════

def list_models_vllm(host: str, port: int,
                     is_https: bool = False,
                     skip_ssl: bool = False,
                     api_key: str = "",
                     timeout: float = TIMEOUT) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = http_get_json(host, port, "/v1/models", is_https, skip_ssl,
                         headers, timeout)
    return [m["id"] for m in data.get("data", [])]


def list_models_ollama(host: str, port: int,
                       is_https: bool = False,
                       skip_ssl: bool = False,
                       timeout: float = TIMEOUT) -> List[str]:
    data = http_get_json(host, port, "/api/tags", is_https, skip_ssl,
                         timeout=timeout)
    return [m["name"] for m in data.get("models", [])]


# ══════════════════════════════════════════════
#  BENCHMARK  ENGINE
# ══════════════════════════════════════════════

class BenchmarkEngine:
    """Runs all benchmark categories against a single endpoint + model."""

    def __init__(self, endpoint_type: str, host: str, port: int,
                 model: str, is_https: bool = False,
                 skip_ssl: bool = False,
                 progress_cb: Optional[Callable] = None,
                 api_key: str = "",
                 context_size: int = 8192):
        self.endpoint_type = endpoint_type  # specific label, e.g. "SGLang"
        self.protocol = endpoint_protocol(endpoint_type)  # "openai" | "ollama"
        self.host = host
        self.port = port
        self.model = model
        self.https = is_https
        self.skip_ssl = skip_ssl
        self.cb = progress_cb or (lambda c, t, m: None)
        self.api_key = api_key
        self.context_size = context_size
        self.results: List[TestResult] = []

    def _stream(self, prompt: str, max_tokens: int):
        if self.protocol == "openai":
            msgs = [{"role": "user", "content": prompt}]
            return _stream_vllm(self.host, self.port, self.model,
                                msgs, max_tokens, self.https, self.skip_ssl,
                                self.api_key)
        return _stream_ollama(self.host, self.port, self.model,
                              prompt, max_tokens, self.https, self.skip_ssl)

    def _token_count(self, meta: Dict) -> int:
        if self.protocol == "ollama":
            return meta.get("eval_count", 0)
        usage = meta.get("usage", {})
        return usage.get("completion_tokens", 0)

    def _run_single(self, prompt: str, max_tokens: int) -> Dict[str, float]:
        t0 = time.perf_counter()
        ttft = 0.0
        tokens = 0
        meta: Dict = {}
        for _text, is_first, m in self._stream(prompt, max_tokens):
            if is_first:
                ttft = time.perf_counter() - t0
            tokens += 1
            meta = m
        total = time.perf_counter() - t0
        api_tokens = self._token_count(meta)
        if api_tokens:
            tokens = api_tokens
        return {"ttft": ttft, "total": total, "tokens": tokens,
                "tps": tokens / total if total > 0 else 0.0}

    def _progress(self, cur: int, total: int, msg: str) -> None:
        self.cb(cur, total, msg)

    def warmup(self) -> None:
        self._progress(0, 1, "Warming up …")
        self._run_single("Hello", 16)
        self._progress(1, 1, "Warm-up done")

    def test_ttft(self) -> None:
        n = 5
        vals: List[float] = []
        for i in range(n):
            self._progress(i + 1, n, f"TTFT run {i + 1}/{n}")
            r = self._run_single("Say hello in one word.", 16)
            vals.append(r["ttft"])
        tr = TestResult("TTFT", "Time to First Token", "s", vals)
        tr.compute_stats()
        self.results.append(tr)

    def test_latency(self) -> None:
        n = 5
        vals: List[float] = []
        for i in range(n):
            self._progress(i + 1, n, f"Latency run {i + 1}/{n}")
            r = self._run_single("Write a short haiku about the sea.", 64)
            vals.append(r["total"])
        tr = TestResult("Latency", "End-to-End Latency", "s", vals)
        tr.compute_stats()
        self.results.append(tr)

    def test_throughput(self) -> None:
        n = 5
        vals: List[float] = []
        for i in range(n):
            self._progress(i + 1, n, f"Throughput run {i + 1}/{n}")
            r = self._run_single(
                "Explain the theory of relativity in detail.", 256)
            vals.append(r["tps"])
        tr = TestResult("Throughput", "Token Throughput", "tok/s", vals)
        tr.compute_stats()
        self.results.append(tr)

    def test_concurrency(self) -> None:
        levels = [1, 2, 4, 8]
        vals: List[float] = []
        for li, level in enumerate(levels):
            self._progress(li + 1, len(levels), f"Concurrency level {level}")
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = [
                    pool.submit(self._run_single,
                                "Say hello in one word.", 16)
                    for _ in range(level)
                ]
                for f in futures:
                    f.result()
            vals.append(time.perf_counter() - t0)
        tr = TestResult("Concurrency", "Wall Time per Level", "s", vals)
        tr.compute_stats()
        tr.name = "Concurrency (1,2,4,8)"
        self.results.append(tr)

    def test_context_scaling(self) -> None:
        # Probe sizes scale up to the user-selected context size.
        max_ctx = max(1024, self.context_size)
        sizes = [max(128, max_ctx // 16), max(256, max_ctx // 4),
                 max(512, max_ctx // 2), max_ctx]
        ttft_vals: List[float] = []
        total_vals: List[float] = []
        tps_vals: List[float] = []
        for si, size in enumerate(sizes):
            self._progress(si + 1, len(sizes), f"Context {size} tokens")
            # Each 27-char filler group is ~7 tokens; leave headroom for
            # the instruction and the generated completion.
            reps = max(1, (size - 64) // 7)
            filler = "lorem ipsum dolor sit amet " * reps
            prompt = (f"Summarise the following text in one sentence:\n\n"
                      f"{filler}")
            r = self._run_single(prompt, 32)
            ttft_vals.append(r["ttft"])
            total_vals.append(r["total"])
            tps_vals.append(r["tps"])
        tr1 = TestResult("Context Scaling", "TTFT vs Context", "s",
                         ttft_vals)
        tr1.compute_stats()
        tr1.name = f"Context TTFT ({', '.join(map(str, sizes))})"
        self.results.append(tr1)
        tr2 = TestResult("Context Scaling", "Total Latency vs Context", "s",
                         total_vals)
        tr2.compute_stats()
        tr2.name = f"Context Latency ({', '.join(map(str, sizes))})"
        self.results.append(tr2)
        tr3 = TestResult("Context Scaling", "Throughput vs Context", "tok/s",
                         tps_vals)
        tr3.compute_stats()
        tr3.name = f"Context Throughput ({', '.join(map(str, sizes))})"
        self.results.append(tr3)

    def test_consistency(self) -> None:
        n = CONSISTENCY_RUNS
        responses: List[str] = []
        for i in range(n):
            self._progress(i + 1, n, f"Consistency run {i + 1}/{n}")
            text = ""
            for tok, _f, _m in self._stream(CONSISTENCY_PROMPT,
                                            CONSISTENCY_MAX_TOKENS):
                text += tok
            responses.append(text.strip().lower())
        unique = len(set(responses))
        tr = TestResult("Consistency", "Unique Responses", "count",
                        [float(unique)])
        tr.compute_stats()
        tr.name = f"Consistency ({n} runs, {unique} unique)"
        self.results.append(tr)

    def run_all(self) -> BenchmarkReport:
        self.warmup()
        self.test_ttft()
        self.test_latency()
        self.test_throughput()
        self.test_concurrency()
        self.test_context_scaling()
        self.test_consistency()
        self._progress(1, 1, "Done")
        return BenchmarkReport(
            endpoint_type=self.endpoint_type,
            endpoint_url=(f"{'https' if self.https else 'http'}://"
                          f"{self.host}:{self.port}"),
            model=self.model,
            timestamp=datetime.now().isoformat(),
            context_size=self.context_size,
            results=self.results,
        )


# ══════════════════════════════════════════════
#  TKINTER  GUI
# ══════════════════════════════════════════════

class App:
    """Main Tkinter application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"LLM Benchmark Tool v{__version__}")
        root.geometry("900x680")
        root.minsize(700, 520)

        self.queue: queue.Queue = queue.Queue()
        self._build_ui()
        self._poll_queue()

    # ── UI construction ──────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ── top frame: endpoint config ──────
        top = ttk.LabelFrame(self.root, text="Endpoint")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Type:").grid(row=0, column=0, sticky="w")
        self.var_type = tk.StringVar(value="Ollama")
        self.cmb_type = ttk.Combobox(
            top, textvariable=self.var_type, width=16, state="readonly",
            values=[label for label, _ in ENDPOINT_TYPES])
        self.cmb_type.grid(row=0, column=1, columnspan=2, sticky="we")
        self.cmb_type.bind("<<ComboboxSelected>>",
                           lambda _e: self._on_type_change())

        ttk.Label(top, text="URL:").grid(row=1, column=0, sticky="w")
        self.var_url = tk.StringVar(value="http://localhost:11434")
        ttk.Entry(top, textvariable=self.var_url, width=40).grid(
            row=1, column=1, columnspan=3, sticky="we")

        ttk.Label(top, text="Model:").grid(row=2, column=0, sticky="w")
        self.var_model = tk.StringVar()
        self.cmb_model = ttk.Combobox(top, textvariable=self.var_model,
                                      width=36, state="readonly")
        self.cmb_model.grid(row=2, column=1, sticky="we")
        ttk.Button(top, text="Refresh",
                   command=self._refresh_models).grid(row=2, column=2)

        ttk.Label(top, text="Context Size:").grid(row=3, column=0,
                                                  sticky="w")
        self.var_context = tk.StringVar(value="8k")
        self.cmb_context = ttk.Combobox(
            top, textvariable=self.var_context, width=12,
            state="readonly",
            values=[label for label, _ in CONTEXT_SIZES])
        self.cmb_context.grid(row=3, column=1, sticky="w")

        self.chk_ssl = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Skip SSL verify",
                        variable=self.chk_ssl).grid(row=4, column=1,
                                                    sticky="w")

        ttk.Label(top, text="API Key:").grid(row=5, column=0, sticky="w")
        self.var_api_key = tk.StringVar()
        self.ent_api_key = ttk.Entry(top, textvariable=self.var_api_key,
                                     width=40, show="*",
                                     state="disabled")
        self.ent_api_key.grid(row=5, column=1, columnspan=3, sticky="we")

        self.btn_connect = ttk.Button(top, text="Connect",
                                      command=self._connect)
        self.btn_connect.grid(row=6, column=1, sticky="w")
        self.lbl_conn = ttk.Label(top, text="Not connected",
                                  foreground="gray")
        self.lbl_conn.grid(row=6, column=2, columnspan=2, sticky="w")

        # ── middle: progress ────────────────
        mid = ttk.LabelFrame(self.root, text="Progress")
        mid.pack(fill="x", **pad)
        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(mid, textvariable=self.var_status).pack(
            side="left", padx=8)
        self.progress = ttk.Progressbar(mid, mode="determinate",
                                        length=300)
        self.progress.pack(side="left", padx=8, fill="x", expand=True)

        # ── bottom: results table ───────────
        bot = ttk.LabelFrame(self.root, text="Results")
        bot.pack(fill="both", expand=True, **pad)

        cols = ("test", "metric", "unit", "mean", "median",
                "min", "max", "stdev", "p95")
        self.tree = ttk.Treeview(bot, columns=cols, show="headings",
                                 height=14)
        headers = {"test": "Test", "metric": "Metric", "unit": "Unit",
                   "mean": "Mean", "median": "Median", "min": "Min",
                   "max": "Max", "stdev": "StDev", "p95": "P95"}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=90, anchor="center")
        self.tree.column("test", width=180, anchor="w")
        self.tree.column("metric", width=160, anchor="w")

        vsb = ttk.Scrollbar(bot, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # ── action buttons ──────────────────
        btns = ttk.Frame(self.root)
        btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Run Benchmark",
                   command=self._start_benchmark).pack(side="left")
        ttk.Button(btns, text="Export CSV",
                   command=self._export_csv).pack(side="left", padx=4)
        ttk.Button(btns, text="Export JSON",
                   command=self._export_json).pack(side="left")
        ttk.Button(btns, text="Open Report",
                   command=self._open_report).pack(side="left", padx=4)
        ttk.Button(btns, text="Export HTML",
                   command=self._export_html).pack(side="left")
        ttk.Separator(btns, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(btns, text="Save Settings",
                   command=self._save_settings).pack(side="left")
        ttk.Button(btns, text="Load Settings",
                   command=self._load_settings).pack(side="left", padx=4)
        ttk.Label(btns, text=f"Runs are logged to {LOG_FILE}",
                  foreground="gray").pack(side="right", padx=4)
        self._last_report: Optional[BenchmarkReport] = None
        self._last_report_path: Optional[str] = None
        self._connected = False
        self._connecting = False
        # Editing the URL invalidates any previous connection.
        self.var_url.trace_add("write", self._on_url_changed)

    def _on_type_change(self) -> None:
        label = self.var_type.get()
        if endpoint_protocol(label) == "openai":
            self.var_url.set(ENDPOINT_DEFAULT_URLS.get(label,
                                                       "http://localhost:8000"))
            self.ent_api_key.configure(state="normal")
        else:
            self.var_url.set("http://localhost:11434")
            self.ent_api_key.configure(state="disabled")
        # The endpoint changed – the user must press Connect again.
        self._reset_connection()

    def _reset_connection(self) -> None:
        """Mark the endpoint as not connected (e.g. after a config change)."""
        self._connected = False
        self._set_conn_status("Not connected", "gray")

    def _on_url_changed(self, *_args: Any) -> None:
        if self._connected:
            self._reset_connection()

    def _set_conn_status(self, text: str, color: str) -> None:
        self.lbl_conn.configure(text=text, foreground=color)

    def _connect(self) -> None:
        """Check the endpoint and discover models (non-blocking)."""
        self._probe_endpoint("Connecting…")

    def _refresh_models(self) -> None:
        """Re-discover the model list (non-blocking)."""
        self._probe_endpoint("Refreshing model list…")

    def _probe_endpoint(self, status_msg: str) -> None:
        """Run a connection check + model discovery in a background thread."""
        if self._connecting:
            return
        host, port, https = parse_url(self.var_url.get())
        skip = self.chk_ssl.get()
        et = self.var_type.get()
        api_key = self.var_api_key.get().strip()
        self._connecting = True
        self._connected = False
        self.btn_connect.configure(state="disabled")
        self._set_conn_status("Connecting…", "gray")
        self.var_status.set(status_msg)

        def _worker() -> None:
            try:
                if endpoint_protocol(et) == "openai":
                    models = list_models_vllm(host, port, https, skip,
                                              api_key,
                                              timeout=CONNECT_TIMEOUT)
                else:
                    models = list_models_ollama(host, port, https, skip,
                                                timeout=CONNECT_TIMEOUT)
                self.queue.put(("connected", models, (host, port, https)))
            except Exception as e:
                self.queue.put(("connect_failed", str(e),
                                (host, port, https)))

        threading.Thread(target=_worker, daemon=True).start()

    def _context_size(self) -> int:
        """Map the selected label (e.g. ``8k``) to its token count."""
        label = self.var_context.get()
        for lbl, val in CONTEXT_SIZES:
            if lbl == label:
                return val
        return DEFAULT_CONTEXT_SIZE

    def _start_benchmark(self) -> None:
        if not self._connected:
            messagebox.showwarning(
                "Warning",
                "Not connected to the endpoint.\nPress Connect first.")
            return
        model = self.var_model.get().strip()
        if not model:
            messagebox.showwarning("Warning", "Please select a model.")
            return
        host, port, https = parse_url(self.var_url.get())
        skip = self.chk_ssl.get()
        et = self.var_type.get()
        api_key = self.var_api_key.get().strip()

        # All user-chosen settings for this run – written to the log.
        settings = {
            "Endpoint Type": et,
            "Endpoint URL": f"{'https' if https else 'http'}://{host}:{port}",
            "Model": model,
            "Context Size": f"{self._context_size()} tokens",
            "Skip SSL Verify": "yes" if skip else "no",
            "API Key": _mask_key(api_key),
        }

        def _progress_cb(cur: int, total: int, msg: str) -> None:
            self.queue.put(("progress", cur, total, msg))

        def _worker() -> None:
            try:
                engine = BenchmarkEngine(et, host, port, model,
                                         https, skip, _progress_cb,
                                         api_key=api_key,
                                         context_size=self._context_size())
                report = engine.run_all()
                html_path = _log_run(report, settings)
                self.queue.put(("done", report, html_path))
            except Exception as e:
                _log_run_failed(settings, str(e))
                self.queue.put(("error", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _current_settings(self) -> Dict[str, str]:
        """Snapshot the current endpoint settings as a dict (for reports)."""
        return {
            "Endpoint Type": self.var_type.get(),
            "Endpoint URL": self.var_url.get().strip(),
            "Model": self.var_model.get().strip(),
            "Context Size": str(self._context_size()),
            "Skip SSL": str(self.chk_ssl.get()),
            "API Key": self.var_api_key.get().strip() or "(none)",
        }

    def _save_settings(self) -> None:
        """Persist current endpoint settings to a JSON file."""
        settings = {
            "endpoint_type": self.var_type.get(),
            "endpoint_url": self.var_url.get().strip(),
            "model": self.var_model.get().strip(),
            "context_size": self._context_size(),
            "skip_ssl": self.chk_ssl.get(),
            "api_key": self.var_api_key.get().strip(),
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            self.var_status.set(f"Settings saved to {SETTINGS_FILE}")
        except OSError as e:
            self.var_status.set(f"Failed to save settings: {e}")

    def _load_settings(self) -> None:
        """Load endpoint settings from the JSON file."""
        if not os.path.isfile(SETTINGS_FILE):
            self.var_status.set("No saved settings found")
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            stored = settings.get("endpoint_type", "Ollama")
            stored = _LEGACY_TYPE_MAP.get(stored, stored)
            if stored not in {label for label, _ in ENDPOINT_TYPES}:
                stored = "Ollama"
            self.var_type.set(stored)
            self.var_url.set(settings.get("endpoint_url", ""))
            self.var_model.set(settings.get("model", ""))
            ctx = int(settings.get("context_size", DEFAULT_CONTEXT_SIZE))
            label = next((lbl for lbl, val in CONTEXT_SIZES if val == ctx),
                         "8k")
            self.var_context.set(label)
            self.chk_ssl.set(settings.get("skip_ssl", False))
            self.var_api_key.set(settings.get("api_key", ""))
            self.ent_api_key.configure(
                state="normal" if endpoint_protocol(self.var_type.get())
                == "openai" else "disabled")
            self._reset_connection()
            self.var_status.set("Settings loaded – press Connect")
        except (json.JSONDecodeError, OSError) as e:
            self.var_status.set(f"Failed to load settings: {e}")

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == "progress":
                    _, cur, total, msg = item
                    self.var_status.set(msg)
                    self.progress["maximum"] = total
                    self.progress["value"] = cur
                elif item[0] == "done":
                    self._last_report = item[1]
                    self._last_report_path = item[2] if len(item) > 2 else None
                    self._populate_results(self._last_report)
                    if self._last_report_path:
                        self.var_status.set(
                            f"Benchmark complete – report: {self._last_report_path}")
                    else:
                        self.var_status.set("Benchmark complete")
                    self.progress["value"] = self.progress["maximum"]
                elif item[0] == "connected":
                    models, target = item[1], item[2]
                    if target != parse_url(self.var_url.get()):
                        continue  # stale – endpoint changed in the meantime
                    self._connecting = False
                    self._connected = True
                    self.btn_connect.configure(state="normal")
                    self.cmb_model["values"] = models
                    if models:
                        self.cmb_model.current(0)
                    self._set_conn_status(
                        f"Connected! ({len(models)} models)", "green")
                    self.var_status.set("Connected")
                elif item[0] == "connect_failed":
                    msg, target = item[1], item[2]
                    if target != parse_url(self.var_url.get()):
                        continue
                    self._connecting = False
                    self._connected = False
                    self.btn_connect.configure(state="normal")
                    if len(msg) > 80:
                        msg = msg[:77] + "…"
                    self._set_conn_status(f"Connection failed: {msg}", "red")
                    self.var_status.set(f"Connection failed: {msg}")
                elif item[0] == "error":
                    self.var_status.set(f"Error: {item[1]}")
                    messagebox.showerror("Benchmark Error", item[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _populate_results(self, report: BenchmarkReport) -> None:
        self.tree.delete(*self.tree.get_children())
        for r in report.results:
            self.tree.insert("", "end", values=(
                r.name, r.metric, r.unit,
                f"{r.mean:.4f}", f"{r.median:.4f}",
                f"{r.min:.4f}", f"{r.max:.4f}",
                f"{r.stdev:.4f}", f"{r.p95:.4f}",
            ))

    def _export_csv(self) -> None:
        if not self._last_report:
            messagebox.showinfo("Info", "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialname=f"benchmark_{self._last_report.model}.csv")
        if path:
            self._last_report.to_csv(path)
            self.var_status.set(f"Exported to {path}")

    def _export_json(self) -> None:
        if not self._last_report:
            messagebox.showinfo("Info", "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialname=f"benchmark_{self._last_report.model}.json")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._last_report.to_dict(), f, indent=2)
            self.var_status.set(f"Exported to {path}")

    def _open_report(self) -> None:
        """Open the most recent HTML report in the default browser."""
        if not self._last_report_path or not os.path.isfile(self._last_report_path):
            messagebox.showinfo("Info", "No HTML report available yet.\n"
                                        "Run a benchmark first.")
            return
        webbrowser.open("file://" + os.path.abspath(self._last_report_path))

    def _export_html(self) -> None:
        """Save the most recent HTML report to a user-chosen location."""
        if not self._last_report:
            messagebox.showinfo("Info", "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            initialname=f"benchmark_{self._last_report.model}.html")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(_generate_html_report(self._last_report,
                                                  self._current_settings()))
                self.var_status.set(f"Exported to {path}")
            except OSError as e:
                messagebox.showerror("Export Error", str(e))


# ══════════════════════════════════════════════
#  ENTRY  POINT
# ══════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    App(root)
    _log_session_start()
    root.mainloop()


if __name__ == "__main__":
    main()