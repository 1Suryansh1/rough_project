"""Shared rich logger for the project."""
from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("scientisttwin")
