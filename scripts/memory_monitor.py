#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory Monitor — Real-time personalization memory debugger
==========================================================

Runs in a separate terminal to monitor all memory-related activities.
Watches the memory directory for file changes and displays:
  - Trace forest index & new trace registrations
  - Embedding updates
  - PersonalizationService event processing logs

Usage:
    python scripts/memory_monitor.py
    python scripts/memory_monitor.py --poll 1.0     # Faster polling
    python scripts/memory_monitor.py --show-content  # Also print file contents on change

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / "DeepTutor.env", override=False)
load_dotenv(project_root / ".env", override=False)

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ── ANSI colors ──────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RESET = "\033[0m"


def _ts() -> str:
    """Current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def _hr(char: str = "─", width: int = 70) -> str:
    return f"{DIM}{char * width}{RESET}"


def _section(title: str) -> None:
    print(f"\n{_hr('━')}")
    print(f"  {BOLD}{title}{RESET}")
    print(_hr("━"))


# ── File state tracker ────────────────────────────────────────────────

class FileState:
    """Track a file's modification time and content hash."""

    def __init__(self, path: Path):
        self.path = path
        self.mtime: float = 0.0
        self.content_hash: str = ""
        self.exists: bool = False

    def check(self) -> bool:
        """Return True if file changed since last check."""
        if not self.path.exists():
            if self.exists:
                self.exists = False
                return True
            return False

        stat = self.path.stat()
        new_mtime = stat.st_mtime

        if new_mtime == self.mtime and self.exists:
            return False

        try:
            content = self.path.read_bytes()
            new_hash = hashlib.md5(content).hexdigest()
        except Exception:
            return False

        changed = (new_hash != self.content_hash) or (not self.exists)
        self.mtime = new_mtime
        self.content_hash = new_hash
        self.exists = True
        return changed

    def read_text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except Exception:
            return ""


# ── Memory directory resolver ─────────────────────────────────────────

def _get_memory_dir() -> Path:
    """Resolve the memory directory path."""
    try:
        from src.services.path_service import get_path_service
        return get_path_service().get_memory_dir()
    except Exception:
        return project_root / "data" / "user" / "workspace" / "memory"


# ── Display helpers ───────────────────────────────────────────────────

def _display_index_summary(index_data: dict[str, Any]) -> None:
    """Display trace forest index summary."""
    traces = index_data.get("traces", [])
    nodes = index_data.get("nodes", [])

    print(f"  Traces: {BOLD}{len(traces)}{RESET}")

    if traces:
        # Show most recent traces
        sorted_traces = sorted(traces, key=lambda t: t.get("timestamp", ""), reverse=True)
        for t in sorted_traces[:5]:
            trace_type = t.get("trace_type", "?")
            trace_id = t.get("trace_id", "?")
            root_text = t.get("root_text", "")[:60]
            icon = f"{CYAN}Q{RESET}" if trace_type == "question" else f"{MAGENTA}S{RESET}"
            print(f"    {icon} {trace_id}  {DIM}{root_text}{RESET}")
        if len(sorted_traces) > 5:
            print(f"    {DIM}... and {len(sorted_traces) - 5} more{RESET}")

    # Node distribution by level
    level_counts: dict[int, int] = {}
    type_counts: dict[str, int] = {}
    for n in nodes:
        lv = n.get("level", 0)
        nt = n.get("node_type", "?")
        level_counts[lv] = level_counts.get(lv, 0) + 1
        type_counts[nt] = type_counts.get(nt, 0) + 1

    print(f"  Nodes:  {BOLD}{len(nodes)}{RESET}", end="")
    if level_counts:
        parts = [f"L{k}={v}" for k, v in sorted(level_counts.items())]
        print(f"  ({', '.join(parts)})", end="")
    print()

    if type_counts:
        parts = [f"{k}={v}" for k, v in sorted(type_counts.items())]
        print(f"  Types:  {DIM}{', '.join(parts)}{RESET}")


def _display_embeddings_summary(embeddings_data: dict[str, Any]) -> None:
    """Display embeddings file summary."""
    count = len(embeddings_data)
    total_dims = 0
    if count > 0:
        first_vec = next(iter(embeddings_data.values()), [])
        total_dims = len(first_vec) if isinstance(first_vec, list) else 0
    print(f"  Vectors: {BOLD}{count}{RESET} nodes embedded", end="")
    if total_dims:
        print(f"  (dim={total_dims})", end="")
    print()


def _display_trace_file(trace_path: Path) -> None:
    """Display summary of a single trace file."""
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        trace_id = data.get("trace_id", trace_path.stem)
        trace_type = data.get("trace_type", "?")
        nodes = data.get("nodes", {})
        root_id = data.get("root_id", "L1")
        root_node = nodes.get(root_id, data.get("root", {}))
        root_text = (root_node.get("text", "") or "")[:80]

        icon = f"{CYAN}Q{RESET}" if trace_type == "question" else f"{MAGENTA}S{RESET}"
        print(f"  {icon} {BOLD}{trace_id}{RESET}")
        print(f"    Root: {root_text}")
        print(f"    Nodes: {len(nodes)}")
        answer_path = data.get("answer_path", "")
        if answer_path:
            print(f"    Answer: {DIM}{answer_path}{RESET}")
    except Exception as exc:
        print(f"  {RED}Error reading {trace_path.name}: {exc}{RESET}")


# ── Event queue monitor ───────────────────────────────────────────────

def _display_event_queue(queue_file: Path, offset_file: Path) -> None:
    """Display event queue status."""
    if not queue_file.exists():
        print(f"  {DIM}Event queue: empty{RESET}")
        return

    total_lines = 0
    try:
        with open(queue_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_lines += 1
    except Exception:
        pass

    offset = 0
    if offset_file.exists():
        try:
            offset = int(offset_file.read_text().strip())
        except (ValueError, IOError):
            pass

    pending = 0
    try:
        with open(queue_file, "r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                if line.strip():
                    pending += 1
    except Exception:
        pass

    status = f"{GREEN}all processed{RESET}" if pending == 0 else f"{YELLOW}{pending} pending{RESET}"
    print(f"  Event queue: {total_lines} total, {status}")


# ── Main monitor loop ─────────────────────────────────────────────────

class MemoryMonitor:
    """Watches the memory directory for changes."""

    def __init__(self, poll_interval: float = 2.0, show_content: bool = False):
        self.poll_interval = poll_interval
        self.show_content = show_content
        self._running = False
        self._memory_dir = _get_memory_dir()
        self._traces_dir = self._memory_dir / "traces"

        # File states
        self._reflection_md = FileState(self._memory_dir / "reflection.md")
        self._summary_md = FileState(self._memory_dir / "memory.md")
        self._weakness_md = FileState(self._memory_dir / "weakness.md")
        self._index_json = FileState(self._traces_dir / "index.json")
        self._embeddings_json = FileState(self._traces_dir / "embeddings.json")
        self._event_queue_file = self._memory_dir / "event_queue" / "events.jsonl"
        self._event_offset_file = self._memory_dir / "event_queue" / "offset.txt"

        # Track known trace files
        self._known_traces: set[str] = set()

    def _scan_trace_files(self) -> set[str]:
        """Get current set of trace JSON files."""
        if not self._traces_dir.exists():
            return set()
        return {
            f.name
            for f in self._traces_dir.iterdir()
            if f.suffix == ".json"
            and f.name not in ("index.json", "embeddings.json")
        }

    def _display_status(self) -> None:
        """Display full current memory status."""
        _section("Memory System Status")
        print(f"  {DIM}Directory: {self._memory_dir}{RESET}")
        print(f"  {DIM}Time: {_ts()}{RESET}")

        # Memory agent documents
        for label, fstate, color in [
            ("reflection.md", self._reflection_md, CYAN),
            ("memory.md", self._summary_md, GREEN),
            ("weakness.md", self._weakness_md, YELLOW),
        ]:
            print(f"\n  {BOLD}[{label}]{RESET}")
            if fstate.path.exists():
                content = fstate.read_text()
                line_count = len(content.splitlines())
                print(f"  Size: {len(content)} chars, {line_count} lines")
                if self.show_content:
                    for line in content.splitlines()[:20]:
                        print(f"  {color}{line}{RESET}")
                    if len(content.splitlines()) > 20:
                        print(f"  {DIM}... ({len(content.splitlines()) - 20} more lines){RESET}")
            else:
                print(f"  {DIM}(not created yet){RESET}")

        # Daily notes
        daily_notes = sorted(self._memory_dir.glob("????-??-??.md"), reverse=True)
        if daily_notes:
            print(f"\n  {BOLD}[Daily Notes]{RESET}")
            for note in daily_notes[:3]:
                size = note.stat().st_size
                print(f"    {note.name}  {DIM}({size} bytes){RESET}")
            if len(daily_notes) > 3:
                print(f"    {DIM}... and {len(daily_notes) - 3} more{RESET}")

        # Trace forest
        print(f"\n  {BOLD}[Trace Forest]{RESET}")
        if self._index_json.path.exists():
            try:
                data = json.loads(self._index_json.read_text())
                _display_index_summary(data)
            except Exception as exc:
                print(f"  {RED}Error: {exc}{RESET}")
        else:
            print(f"  {DIM}(no traces yet){RESET}")

        # Embeddings
        print(f"\n  {BOLD}[Embeddings]{RESET}")
        if self._embeddings_json.path.exists():
            try:
                data = json.loads(self._embeddings_json.read_text())
                _display_embeddings_summary(data)
            except Exception as exc:
                print(f"  {RED}Error: {exc}{RESET}")
        else:
            print(f"  {DIM}(no embeddings yet){RESET}")

        # Event queue
        print(f"\n  {BOLD}[Event Queue]{RESET}")
        _display_event_queue(self._event_queue_file, self._event_offset_file)

        print()

    def _check_changes(self) -> None:
        """Check for file changes and report them."""
        ts = _ts()

        # Memory agent document changes
        for label, fstate, color in [
            ("reflection.md", self._reflection_md, CYAN),
            ("memory.md", self._summary_md, GREEN),
            ("weakness.md", self._weakness_md, YELLOW),
        ]:
            if fstate.check():
                content = fstate.read_text()
                line_count = len(content.splitlines())
                print(f"  {ts} {color}[{label.upper()} UPDATED]{RESET}  {line_count} lines, {len(content)} chars")
                if self.show_content:
                    for line in content.splitlines()[-10:]:
                        print(f"    {color}{line}{RESET}")

        # Index changes
        if self._index_json.check():
            try:
                data = json.loads(self._index_json.read_text())
                traces_count = len(data.get("traces", []))
                nodes_count = len(data.get("nodes", []))
                print(
                    f"  {ts} {CYAN}[INDEX UPDATED]{RESET}"
                    f"  traces={traces_count}, nodes={nodes_count}"
                )
            except Exception:
                print(f"  {ts} {CYAN}[INDEX UPDATED]{RESET}")

        # Embeddings changes
        if self._embeddings_json.check():
            try:
                data = json.loads(self._embeddings_json.read_text())
                print(f"  {ts} {MAGENTA}[EMBEDDINGS UPDATED]{RESET}  {len(data)} vectors")
            except Exception:
                print(f"  {ts} {MAGENTA}[EMBEDDINGS UPDATED]{RESET}")

        # New trace files
        current_traces = self._scan_trace_files()
        new_traces = current_traces - self._known_traces
        for trace_name in sorted(new_traces):
            trace_path = self._traces_dir / trace_name
            print(f"  {ts} {YELLOW}[NEW TRACE]{RESET}  {trace_name}")
            if self.show_content:
                _display_trace_file(trace_path)
        self._known_traces = current_traces

        # Daily notes (check for new files)
        # This is handled implicitly by the full status display

    def run(self) -> None:
        """Main monitor loop."""
        self._running = True

        # Initial state
        self._reflection_md.check()
        self._summary_md.check()
        self._weakness_md.check()
        self._index_json.check()
        self._embeddings_json.check()
        self._known_traces = self._scan_trace_files()

        self._display_status()

        print(f"  {BOLD}Watching for changes...{RESET}  (poll every {self.poll_interval}s)")
        print(_hr())

        while self._running:
            try:
                time.sleep(self.poll_interval)
                self._check_changes()
            except KeyboardInterrupt:
                break

    def stop(self) -> None:
        self._running = False


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory Monitor — real-time personalization memory debugger",
    )
    parser.add_argument(
        "--poll", type=float, default=2.0,
        help="Polling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--show-content", action="store_true",
        help="Show full file contents on change",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  {BOLD}{CYAN}DeepTutor Memory Monitor{RESET}")
    print(f"{'=' * 70}")
    print(f"  {DIM}Press Ctrl+C to stop{RESET}")

    monitor = MemoryMonitor(
        poll_interval=args.poll,
        show_content=args.show_content,
    )

    def signal_handler(sig, frame):
        monitor.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor.run()

    print(f"\n{'=' * 70}")
    print(f"  {DIM}Memory Monitor stopped.{RESET}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
