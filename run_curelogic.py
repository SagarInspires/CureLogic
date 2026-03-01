#!/usr/bin/env python3
"""
CureLogic — System Launcher
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
ML   = ROOT / 'ml'
BE   = ROOT / 'backend'

CYAN   = '\033[96m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
BOLD   = '\033[1m'
RESET  = '\033[0m'


# ──────────────────────────────────────────────
# UI Helpers
# ──────────────────────────────────────────────

def banner():
    print(f"""{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════╗
║         CureLogic — AI Precast Cycle Optimizer           ║
║         L&T CreaTech Hackathon | Problem Statement 01    ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {CYAN}→{RESET}  {msg}")


# ──────────────────────────────────────────────
# Dependency Check
# ──────────────────────────────────────────────

def check_deps():
    print(f"\n{BOLD}[1/4] Checking dependencies...{RESET}")
    required = ['flask', 'sklearn', 'numpy', 'pandas', 'joblib', 'scipy']
    missing = []

    for pkg in required:
        try:
            __import__('sklearn' if pkg == 'sklearn' else pkg)
            ok(pkg)
        except ImportError:
            warn(f"{pkg} missing — will install")
            missing.append(pkg)

    if missing:
        pip_names = {'sklearn': 'scikit-learn'}
        install_names = [pip_names.get(p, p) for p in missing]

        info(f"Installing: {', '.join(install_names)}")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install'] + install_names
        )
        ok("Dependencies installed")


# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────

def train_models():
    print(f"\n{BOLD}[2/4] Checking ML models...{RESET}")

    strength_path = ML / 'model_strength.pkl'
    time_path     = ML / 'model_time.pkl'

    if strength_path.exists() and time_path.exists():
        ok("Models found")
        return

    warn("Models not found — training now...")

    if not (ML / 'features_engineered.csv').exists():
        info("Running EDA...")
        subprocess.check_call([sys.executable, str(ML / '01_eda.py')])

    info("Training SVM models...")
    subprocess.check_call([sys.executable, str(ML / '02_svm_model.py')])

    if strength_path.exists():
        ok("Models trained successfully")
    else:
        err("Model training failed")
        sys.exit(1)


# ──────────────────────────────────────────────
# Dashboard Preparation (FIXED UTF-8 BUG HERE)
# ──────────────────────────────────────────────

def prepare_dashboard(port):
    print(f"\n{BOLD}[3/4] Preparing dashboard...{RESET}")

    src = BE / 'sagar_dashboard_v2.html'

    if not src.exists():
        warn("Dashboard file not found")
        return

    try:
        # FIX: Explicit UTF-8 encoding
        with open(src, 'r', encoding='utf-8') as f:
            html = f.read()

        html = html.replace(
            'http://192.168.1.100:5000',
            f'http://localhost:{port}'
        )

        # Write back safely in UTF-8
        with open(src, 'w', encoding='utf-8') as f:
            f.write(html)

        ok("Dashboard ready")

    except Exception as e:
        err(f"Dashboard preparation failed: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────
# Backend Launch
# ──────────────────────────────────────────────

def start_backend(port, open_browser):
    print(f"\n{BOLD}[4/4] Starting Backend...{RESET}\n")

    print(f"{CYAN}{'─'*50}{RESET}")
    print(f"Dashboard  →  {GREEN}http://localhost:{port}/{RESET}")
    print(f"Predict    →  http://localhost:{port}/api/predict")
    print(f"Status     →  http://localhost:{port}/api/status")
    print(f"{CYAN}{'─'*50}{RESET}")
    print(f"\nPress Ctrl+C to stop\n")

    if open_browser:
        import threading
        def _open():
            time.sleep(2.5)
            webbrowser.open(f'http://localhost:{port}/')
        threading.Thread(target=_open, daemon=True).start()
        ok("Browser will open automatically")

    env = os.environ.copy()
    env['FLASK_PORT'] = str(port)
    env['FLASK_ENV']  = 'production'

    os.chdir(str(BE))

    try:
        subprocess.run(
            [sys.executable, 'app.py'],
            env=env
        )
    except KeyboardInterrupt:
        print(f"\n{YELLOW}CureLogic shutting down gracefully.{RESET}\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    banner()
    check_deps()
    train_models()
    prepare_dashboard(args.port)
    start_backend(args.port, not args.no_browser)
    