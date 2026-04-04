import sys
import os
import json
import time
import re
import subprocess
import argparse
import glob
import threading
from pathlib import Path
from datetime import datetime
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.align import Align

# [SENTINEL-WATCHDOG] Ensure backend package is discoverable
sys.path.append(str(Path(__file__).parent.parent.parent))

console = Console()

class OmegaSentinel:
    def __init__(self, target_cmd: str):
        self.target_cmd = target_cmd
        self.console = Console() # Initialize instance console
        self.pulse_dir = Path("evaluation/latest/pulses")
        self.start_time = datetime.now()
        self.max_retries = 3
        self.log_tail = []
        
        # v4.8 Clean Sweep
        if self.pulse_dir.exists():
            for f in self.pulse_dir.glob("*.json"):
                try: os.remove(f)
                except Exception: pass
        else:
            self.pulse_dir.mkdir(parents=True, exist_ok=True)

    def reader_thread(self, process):
        """Thread to capture logs without blocking the UI."""
        for line in process.stdout:
            try:
                line = line.strip()
                if line:
                    clean_line = re.sub(r'\[.*?\]', '', line).strip()
                    if clean_line:
                        self.log_tail.append(clean_line)
                        if len(self.log_tail) > 100: self.log_tail.pop(0)
            except Exception: pass

    def generate_ui(self):
        # 0. Global Metrics Calculation
        p_files = list(self.pulse_dir.glob("*.json"))
        accuracies = []
        crunching_count = 0
        for pf in p_files:
            try:
                with open(pf, "r") as f:
                    d = json.load(f)
                    if d.get("acc", 0) > 0: accuracies.append(d.get("acc", 0))
                    if "Crunching" in d.get("status", ""): crunching_count += 1
            except Exception: pass
        
        mastery = (sum(accuracies) / len(accuracies) * 100) if accuracies else 0.0
        xeon_pressure = (crunching_count / 10) * 100
        runtime = str(datetime.now() - self.start_time).split(".")[0]

        # 1. Header
        header = Panel(
            Align.center(
                Text.assemble(
                    (" ◢◤ ", "bold magenta"),
                    ("HYBRID QCNN RESEARCH OVERDRIVE - v6.0 BUILD ", "bold white"),
                    ("◢◤ ", "bold magenta"),
                    ("\n", ""),
                    (f"MASTER RUNTIME: {runtime} | ", "cyan"),
                    (f"GLOBAL MASTERY: {mastery:4.1f}% | ", "bold green"),
                    (f"XEON PRESSURE: {xeon_pressure:3.0f}%", "bold yellow")
                )
            ),
            box=box.DOUBLE, border_style="bold magenta"
        )

        # 2. Expert Grid
        matrix_table = Table(box=box.ROUNDED, expand=True, border_style="dim cyan")
        matrix_table.add_column("Expert Seed", style="bold cyan")
        matrix_table.add_column("Model Variant", style="magenta")
        matrix_table.add_column("Epoch", justify="center")
        matrix_table.add_column("Progress Bar", justify="left", ratio=2)
        matrix_table.add_column("Batch", justify="center")
        matrix_table.add_column("Live Metrics", justify="center")
        matrix_table.add_column("Engine Status", justify="center")

        active_pulses = sorted(p_files, key=os.path.getmtime, reverse=True)
        for p_file in active_pulses:
            try:
                name_parts = p_file.stem.split("_")
                lang = name_parts[0].capitalize()
                seed_info = name_parts[-1].upper() if len(name_parts) >= 2 else "S??"
                model_type = "_".join(name_parts[1:-1]).upper() if len(name_parts) >= 3 else "QCNN"
                
                with open(p_file, "r") as f:
                    data = json.load(f)
                
                final_model = data.get("model", model_type).upper()
                epoch, batch, total = data.get("epoch", 0), data.get("batch", 0), data.get("total_batches", 1)
                status = data.get("status", "Crunching").split(".")[0]
                acc, loss = data.get("acc", 0.0) * 100, data.get("loss", 0.0)
                
                filled = int((batch / total) * 20) if total > 0 else 0
                bar = f"[bold green]{'#' * filled}[/][dim]{'.' * (20 - filled)}[/] [cyan]{(batch/total*100 if total>0 else 0):3.0f}%[/]"
                metrics = f"[bold green]{acc:5.1f}%[/] [dim]L:{loss:4.2f}[/]"
                
                if time.time() - data.get("timestamp", 0) > 180:
                    status = "[bold yellow]QUEUED[/]" if batch == 0 else "[bold red]STALLED[/]"
                matrix_table.add_row(f"{lang}-{seed_info}", final_model, str(epoch), bar, f"{batch}/{total}", metrics, status)
            except Exception: pass

        # 3. Footer
        pulse_text = "\n".join(self.log_tail[-10:])
        footer_left = Panel(Text(pulse_text, style="dim white"), title="Neural Pulse (Deepstream IPC)", border_style="cyan")
        
        telemetry = [
            f"[bold cyan]ENGINE:[/][bold white] v6.0 LOCKED[/]",
            f"[bold cyan]GPU:[/][bold white] QUADRO P2200[/]",
            f"[bold cyan]CORES:[/][bold white] 32 XEON[/]",
            f"[bold cyan]PID:[/][bold green] {os.getpid()}[/]",
            f"[bold cyan]MODE:[/][bold yellow] MULTI-EXPERT[/]"
        ]
        footer_right = Panel("\n".join(telemetry), title="SOTA Global State", border_style="magenta")

        # 4. Layout
        layout = Layout()
        layout.split_column(
            Layout(header, size=4),
            Layout(Panel(matrix_table, title="Multilingual Expert Matrix", border_style="cyan")),
            Layout(name="footer", size=12)
        )
        layout["footer"].split_row(
            Layout(footer_left, ratio=2),
            Layout(footer_right, ratio=1)
        )
        return layout

    def run(self):
        # 0. Engagment Prep
        self.console.clear()
        self.console.print("[bold green]Initializing Cinematic Engine v6.0 Build...[/]")
        time.sleep(1.0) # Settle buffer
        
        # 1. Engage Overdrive
        with Live(self.generate_ui(), refresh_per_second=4, screen=True, auto_refresh=True) as live:
            attempt = 0
            while attempt < self.max_retries:
                env = os.environ.copy()
                env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})

                process = subprocess.Popen(
                    self.target_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env
                )
                
                # Start Threaded Reader
                t = threading.Thread(target=self.reader_thread, args=(process,), daemon=True)
                t.start()
                
                while process.poll() is None:
                    live.update(self.generate_ui())
                    time.sleep(0.2)
                
                if process.returncode == 0:
                    live.update(self.generate_ui())
                    break
                attempt += 1
                time.sleep(5)

if __name__ == "__main__":
    python_exe = sys.executable
    cmd = f'set PYTHONUNBUFFERED=1 && "{python_exe}" backend/training/orchestrator.py --mode matrix'
    sentinel = OmegaSentinel(cmd)
    sentinel.run()
