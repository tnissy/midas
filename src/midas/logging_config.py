"""Centralized logging configuration for Midas."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from midas.config import DATA_DIR

# =============================================================================
# Constants
# =============================================================================

LOG_DIR = DATA_DIR / "logs"
AGENT_LOG_DIR = LOG_DIR / "agents"
RUN_LOG_DIR = LOG_DIR / "runs"

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log formats
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# =============================================================================
# Logger Registry
# =============================================================================

_loggers: dict[str, logging.Logger] = {}
_main_logger: Optional[logging.Logger] = None
_current_run_id: Optional[str] = None


def get_run_id() -> str:
    """Get or create a run ID for this session."""
    global _current_run_id
    if _current_run_id is None:
        _current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _current_run_id


def reset_run_id() -> str:
    """Reset the run ID (call at start of new CLI command)."""
    global _current_run_id
    _current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _current_run_id


# =============================================================================
# Main Logger Setup
# =============================================================================


def setup_main_logger(level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Set up the main Midas logger.

    This logger writes to:
    - Console (INFO and above)
    - data/logs/midas.log (all levels)
    - data/logs/runs/run_<timestamp>.log (all levels)

    Returns:
        The main Midas logger
    """
    global _main_logger

    if _main_logger is not None:
        return _main_logger

    logger = logging.getLogger("midas")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Main log file handler (append mode)
    main_log_path = LOG_DIR / "midas.log"
    main_file_handler = logging.FileHandler(main_log_path, encoding="utf-8")
    main_file_handler.setLevel(logging.DEBUG)
    main_file_formatter = logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
    main_file_handler.setFormatter(main_file_formatter)
    logger.addHandler(main_file_handler)

    # Run-specific log file handler
    run_id = get_run_id()
    run_log_path = RUN_LOG_DIR / f"run_{run_id}.log"
    run_file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    run_file_handler.setLevel(logging.DEBUG)
    run_file_handler.setFormatter(main_file_formatter)
    logger.addHandler(run_file_handler)

    _main_logger = logger
    logger.info(f"=== Midas Session Started (Run ID: {run_id}) ===")

    return logger


def get_main_logger() -> logging.Logger:
    """Get the main Midas logger, setting it up if needed."""
    if _main_logger is None:
        return setup_main_logger()
    return _main_logger


# =============================================================================
# Agent Logger Setup
# =============================================================================


def get_agent_logger(
    agent_name: str,
    suffix: Optional[str] = None,
    level: int = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """Get or create a logger for a specific agent.

    Args:
        agent_name: Name of the agent (e.g., "company_watcher", "foresight_manager")
        suffix: Optional suffix for the log file (e.g., ticker symbol)
        level: Console log level

    Returns:
        Configured logger for the agent
    """
    # Create unique logger name
    logger_key = f"{agent_name}_{suffix}" if suffix else agent_name

    if logger_key in _loggers:
        return _loggers[logger_key]

    # Create logger as child of main logger
    logger = logging.getLogger(f"midas.{agent_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = True  # Propagate to main logger

    # Agent-specific log file
    run_id = get_run_id()
    if suffix:
        log_filename = f"{agent_name}_{suffix}_{run_id}.log"
    else:
        log_filename = f"{agent_name}_{run_id}.log"

    log_path = AGENT_LOG_DIR / log_filename

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    _loggers[logger_key] = logger

    logger.info(f"=== {agent_name} Started ===")
    logger.debug(f"Log file: {log_path}")

    return logger


# =============================================================================
# Utility Functions
# =============================================================================


def log_section(logger: logging.Logger, title: str, char: str = "=", width: int = 60):
    """Log a section header."""
    line = char * width
    logger.info(line)
    logger.info(f" {title}")
    logger.info(line)


def log_step(logger: logging.Logger, step: int, total: int, message: str):
    """Log a step in a multi-step process."""
    logger.info(f"[{step}/{total}] {message}")


def log_result(logger: logging.Logger, success: bool, message: str):
    """Log a result (success or failure)."""
    if success:
        logger.info(f"OK: {message}")
    else:
        logger.warning(f"FAILED: {message}")


def log_data_summary(logger: logging.Logger, data_name: str, count: int, details: Optional[str] = None):
    """Log a data summary."""
    msg = f"{data_name}: {count} items"
    if details:
        msg += f" ({details})"
    logger.info(msg)


# =============================================================================
# LangGraph Logging Helpers
# =============================================================================


def log_agent_start(logger: logging.Logger, agent_name: str, initial_state: dict | None = None):
    """Log the start of a LangGraph agent execution.

    Args:
        logger: The logger to use
        agent_name: Name of the agent
        initial_state: Optional initial state to log
    """
    logger.info("=" * 80)
    logger.info(f"AGENT START: {agent_name}")
    logger.info("=" * 80)
    if initial_state:
        logger.debug(f"Initial state: {initial_state}")


def log_agent_end(logger: logging.Logger, agent_name: str, final_state: dict | None = None, error: str | None = None):
    """Log the end of a LangGraph agent execution.

    Args:
        logger: The logger to use
        agent_name: Name of the agent
        final_state: Optional final state to log
        error: Optional error message if the agent failed
    """
    logger.info("=" * 80)
    if error:
        logger.error(f"AGENT END: {agent_name} (FAILED: {error})")
    else:
        logger.info(f"AGENT END: {agent_name} (SUCCESS)")
    logger.info("=" * 80)
    if final_state:
        logger.debug(f"Final state: {final_state}")


def log_node_start(logger: logging.Logger, node_name: str, state: dict | None = None):
    """Log the start of a graph node execution.

    Args:
        logger: The logger to use
        node_name: Name of the node
        state: Optional current state to log
    """
    logger.info("-" * 60)
    logger.info(f"NODE START: {node_name}")
    logger.info("-" * 60)
    if state:
        logger.debug(f"Node input state: {state}")


def log_node_end(logger: logging.Logger, node_name: str, state: dict | None = None):
    """Log the end of a graph node execution.

    Args:
        logger: The logger to use
        node_name: Name of the node
        state: Optional output state to log
    """
    logger.info(f"NODE END: {node_name}")
    if state:
        logger.debug(f"Node output state: {state}")


def log_transition(logger: logging.Logger, from_node: str, to_node: str, condition: str | None = None):
    """Log a graph transition between nodes.

    Args:
        logger: The logger to use
        from_node: Source node name
        to_node: Destination node name
        condition: Optional condition that triggered this transition
    """
    if condition:
        logger.info(f"TRANSITION: {from_node} -> {to_node} (condition: {condition})")
    else:
        logger.info(f"TRANSITION: {from_node} -> {to_node}")


def log_state_change(logger: logging.Logger, key: str, old_value: any, new_value: any):
    """Log a state change.

    Args:
        logger: The logger to use
        key: State key that changed
        old_value: Previous value
        new_value: New value
    """
    logger.debug(f"STATE CHANGE: {key}: {old_value} -> {new_value}")


# =============================================================================
# Cleanup
# =============================================================================


def cleanup_loggers():
    """Clean up all loggers (call at end of session)."""
    global _loggers, _main_logger, _current_run_id

    for logger in _loggers.values():
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if _main_logger:
        _main_logger.info("=== Midas Session Ended ===")
        for handler in _main_logger.handlers[:]:
            handler.close()
            _main_logger.removeHandler(handler)

    _loggers.clear()
    _main_logger = None
    _current_run_id = None
