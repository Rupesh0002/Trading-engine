"""
Centralised logging setup.
Log level and file paths are read from config/settings.py → .env.

Console output is colour-coded:
  DEBUG   → cyan
  INFO    → white/default
  WARNING → yellow  (bold)
  ERROR   → red     (bold)
  CRITICAL→ red background
"""
from __future__ import annotations

import logging
import os
import sys

from config.settings import ERROR_LOG_FILE, LOG_LEVEL

_configured = False

# ANSI colour codes — applied only when the terminal supports colour
_RESET  = "\033[0m"
_COLOURS = {
    "DEBUG":    "\033[36m",        # cyan
    "INFO":     "",                # default terminal colour
    "WARNING":  "\033[1;33m",      # bold yellow
    "ERROR":    "\033[1;31m",      # bold red
    "CRITICAL": "\033[1;41m",      # bold white on red background
}


class _ColourFormatter(logging.Formatter):
    """Formatter that wraps the level name (and full line for WARNING+) in colour."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        msg = super().format(record)
        if not colour:
            return msg
        # Colour the entire line for WARNING / ERROR / CRITICAL so they stand out
        if record.levelno >= logging.WARNING:
            return f"{colour}{msg}{_RESET}"
        # For DEBUG: only colour the level tag
        return msg.replace(record.levelname, f"{colour}{record.levelname}{_RESET}", 1)


def _supports_colour() -> bool:
    """Return True when stdout is a real terminal that accepts ANSI codes."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _setup() -> None:
    global _configured
    if _configured:
        return

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    plain_fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt   = "%H:%M:%S"

    if _supports_colour():
        console_formatter = _ColourFormatter(fmt=plain_fmt, datefmt=datefmt)
    else:
        console_formatter = logging.Formatter(fmt=plain_fmt, datefmt=datefmt)

    # Console handler — colours when running in a real terminal
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(console_formatter)

    # File handler — plain text, WARNING and above only
    os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(ERROR_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(fmt=plain_fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _setup()
    return logging.getLogger(name)
