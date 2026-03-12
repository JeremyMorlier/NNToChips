import os
from pathlib import Path


def mkdir_recursive(directory: str) -> None:
    """
    Recursively create directories if they don't exist.

    Args:
        directory: Path to the directory to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def ensure_dir_exists(file_path: str) -> None:
    """
    Create parent directories for a file path if they don't exist.

    Args:
        file_path: Path to the file
    """
    directory = os.path.dirname(file_path)
    if directory:
        mkdir_recursive(directory)
