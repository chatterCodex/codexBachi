import pickle
from typing import Any

def save_results(results: Any, file_path: str) -> None:
    """Saves results to a file using pickle.

    Args:
        results (Any): The results to save.
        file_path (str): The path to the file where results will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)


def load_results(file_path: str) -> Any:
    """Loads results from a file using pickle.

    Args:
        file_path (str): The path to the file from which results will be loaded.

    Returns:
        Any: The loaded results.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)