"""
load_model tool for sktime MCP.

Restores a previously saved sktime estimator from a path or URI
and registers it into the handle manager.
"""

from typing import Any, Dict

from sktime_mcp.runtime.handles import get_handle_manager


def load_model_tool(path: str) -> Dict[str, Any]:
    """
    Load a saved sktime estimator from a path or URI and register it as a handle.

    Args:
        path: Directory path or MLflow URI from which to load the model.

    Returns:
        Dictionary with success status, handle, and model metadata.
    """
    try:
        from sktime.utils.mlflow_sktime import load_model
    except ImportError:
        return {
            "success": False,
            "error": (
                "Could not import sktime.utils.mlflow_sktime.load_model. "
                "Please ensure sktime is installed with MLflow support: "
                "pip install sktime[mlflow]"
            ),
        }

    try:
        estimator = load_model(path)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model from '{path}': {str(e)}",
            "path": path,
        }

    estimator_class = type(estimator).__name__
    estimator_module = type(estimator).__module__

    metadata = {
        "loaded_from": path,
        "class": estimator_class,
        "module": estimator_module,
    }

    handle_manager = get_handle_manager()
    handle_id = handle_manager.create_handle(
        estimator_name=estimator_class,
        instance=estimator,
        params={},
        metadata=metadata,
    )
    handle_manager.mark_fitted(handle_id)

    return {
        "success": True,
        "handle": handle_id,
        "estimator": estimator_class,
        "path": path,
        "fitted": True,
        "metadata": metadata,
    }
