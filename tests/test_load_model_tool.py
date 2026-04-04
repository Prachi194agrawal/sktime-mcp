"""Tests for load_model tool."""

import sys
import types

sys.path.insert(0, "src")

from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.load_model import load_model_tool


class DummyLoadedEstimator:
    """Simple estimator-like object for load tests."""


def test_load_model_registers_and_marks_fitted(monkeypatch):
    """load_model_tool should register loaded estimator and mark it fitted."""

    def _fake_load_model(path):
        assert path == "/tmp/model-dir"
        return DummyLoadedEstimator()

    fake_module = types.ModuleType("sktime.utils.mlflow_sktime")
    fake_module.load_model = _fake_load_model
    monkeypatch.setitem(sys.modules, "sktime.utils.mlflow_sktime", fake_module)

    handle_manager = get_handle_manager()
    handle_manager.clear_all()

    result = load_model_tool("/tmp/model-dir")

    assert result["success"] is True
    assert result["handle"].startswith("est_")
    assert result["estimator"] == "DummyLoadedEstimator"
    assert result["fitted"] is True
    assert result["metadata"]["loaded_from"] == "/tmp/model-dir"

    assert handle_manager.exists(result["handle"])
    assert handle_manager.is_fitted(result["handle"])


def test_load_model_returns_error_when_loading_fails(monkeypatch):
    """load_model_tool should return a clear error when deserialization fails."""

    def _fake_load_model(_path):
        raise RuntimeError("bad artifact")

    fake_module = types.ModuleType("sktime.utils.mlflow_sktime")
    fake_module.load_model = _fake_load_model
    monkeypatch.setitem(sys.modules, "sktime.utils.mlflow_sktime", fake_module)

    result = load_model_tool("mlflow-artifacts:/run/model")

    assert result["success"] is False
    assert "Failed to load model" in result["error"]
    assert result["path"] == "mlflow-artifacts:/run/model"


def test_load_model_returns_error_when_import_unavailable(monkeypatch):
    """load_model_tool should return guidance if mlflow_sktime cannot be imported."""

    monkeypatch.delitem(sys.modules, "sktime.utils.mlflow_sktime", raising=False)

    import builtins

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sktime.utils.mlflow_sktime":
            raise ImportError("not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    result = load_model_tool("/tmp/model")

    assert result["success"] is False
    assert "pip install sktime[mlflow]" in result["error"]
