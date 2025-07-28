import pickle
import re
from typing import Any
import json
import pandas as pd
import inspect

def save_results(results: Any, file_path: str) -> None:
    """Saves results to a file using pickle.

    Args:
        results (Any): The results to save.
        file_path (str): The path to the file where results will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

def inspect_classes(results: Any) -> None:
    if isinstance(results, (tuple, set)):
        print(f"Results is a {type(results).__name__}")
        for i, item in enumerate(results):
            print(f"[{i}] Class: {item.__class__.__name__}, Module: {item.__class__.__module__}")
    elif isinstance(results, pd.DataFrame):
        print("Type: pandas.DataFrame")
    else:
        print(f"Unexpected type: {type(results)}")

def inspect_deep_nested_lists(results):
    if isinstance(results, tuple):
        for i, part in enumerate(results):
            if isinstance(part, list):
                print(f"\nTuple[{i}] is a list with {len(part)} items.")
                for j, item in enumerate(part[:3]):
                    print(f"  [{j}] Class: {item.__class__.__name__}, from: {item.__class__.__module__}")
                    if isinstance(item, list):
                        for k, sub_item in enumerate(item[:3]):
                            print(f"    [{j}][{k}] → {sub_item.__class__.__name__} from {sub_item.__class__.__module__}")


def save_results_for_humans(results: Any, file_path: str) -> None:

    if isinstance(results, pd.DataFrame):
        results.to_csv(file_path + '.csv', index=False)

    elif isinstance(results, (list, tuple, set)):
        for i, item in enumerate(results):
            out_path = f"{file_path}_part{i}.json"
            if hasattr(item, "to_dict"):
                with open(out_path, 'w') as f:
                    json.dump(item.to_dict(), f, indent=2, default=str)
            elif isinstance(item, pd.DataFrame):
                item.to_csv(out_path.replace(".json", ".csv"), index=False)
            elif isinstance(item, list) and all(hasattr(x, "to_dict") for x in item):
                with open(out_path, 'w') as f:
                    json.dump([x.to_dict() for x in item], f, indent=2, default=str)
            else:
                with open(out_path, 'w') as f:
                    json.dump(item, f, indent=2, default=str)

    elif hasattr(results, "to_dict"):
        with open(file_path + '.json', 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

    else:
        raise ValueError(f"Unsupported results type for human-readable format: {type(results)}")


def load_results(file_path: str) -> Any:
    """Loads results from a file using pickle.

    Args:
        file_path (str): The path to the file from which results will be loaded.

    Returns:
        Any: The loaded results.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)