# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FeastClient."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

# Test utilities (local to avoid cross-module dependency on kubeflow.trainer.test)
SUCCESS = "success"
FAILED = "Failed"


@dataclass
class TestCase:
    name: str
    expected_status: str = SUCCESS
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None
    expected_error: type[Exception] | None = None
    __test__ = False


@pytest.fixture(autouse=True)
def skip_if_no_feast():
    """Skip tests if feast not installed."""
    pytest.importorskip("feast")


@pytest.fixture
def mock_feast_store():
    """Create a mock FeatureStore."""
    return MagicMock()


@pytest.fixture
def client(mock_feast_store, monkeypatch):
    """Create FeastClient with mock FeatureStore."""
    from kubeflow.feast.api.feast_client import FeastClient

    monkeypatch.setattr("feast.FeatureStore", lambda **kwargs: mock_feast_store)

    return FeastClient()


def test_init_import_error(monkeypatch):
    """Test that __init__ raises helpful ImportError when feast missing."""
    import sys

    monkeypatch.setitem(sys.modules, "feast", None)

    with pytest.raises(ImportError, match="pip install 'kubeflow\\[feast\\]'"):
        from kubeflow.feast.api import feast_client

        importlib = pytest.importorskip("importlib")
        importlib.reload(feast_client)
        feast_client.FeastClient()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default initialization with no args",
            expected_status=SUCCESS,
            config={},
            expected_output={"repo_path": None},
        ),
        TestCase(
            name="initialization with repo_path",
            expected_status=SUCCESS,
            config={"repo_path": "/tmp/feast-repo"},
            expected_output={"repo_path": "/tmp/feast-repo"},
        ),
    ],
)
def test_init(test_case, monkeypatch):
    """Test FeastClient initialization with repo_path."""
    from kubeflow.feast.api.feast_client import FeastClient

    mock_feast_store_class = MagicMock()
    mock_feast_store_instance = MagicMock()
    mock_feast_store_class.return_value = mock_feast_store_instance

    monkeypatch.setattr("feast.FeatureStore", mock_feast_store_class)

    client = FeastClient(**test_case.config)

    mock_feast_store_class.assert_called_once_with(**test_case.expected_output)
    assert client._feature_store == mock_feast_store_instance


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="initialization with config dict",
            expected_status=SUCCESS,
            config={"config": {"project": "test"}},
        ),
        TestCase(
            name="config takes precedence over repo_path",
            expected_status=SUCCESS,
            config={"repo_path": "/tmp/feast-repo", "config": {"project": "test"}},
        ),
    ],
)
def test_init_with_config(test_case, monkeypatch):
    """Test FeastClient initialization converts config dict to RepoConfig."""
    from kubeflow.feast.api.feast_client import FeastClient

    mock_feast_store_class = MagicMock()
    mock_feast_store_instance = MagicMock()
    mock_feast_store_class.return_value = mock_feast_store_instance
    mock_repo_config_class = MagicMock()
    mock_repo_config_instance = MagicMock()
    mock_repo_config_class.return_value = mock_repo_config_instance

    monkeypatch.setattr("feast.FeatureStore", mock_feast_store_class)
    monkeypatch.setattr("feast.RepoConfig", mock_repo_config_class)

    client = FeastClient(**test_case.config)

    mock_repo_config_class.assert_called_once_with(**test_case.config["config"])
    mock_feast_store_class.assert_called_once_with(config=mock_repo_config_instance)
    assert client._feature_store == mock_feast_store_instance


def test_feature_store_property(client, mock_feast_store):
    """Test feature_store property provides access to underlying FeatureStore."""
    assert client.feature_store == mock_feast_store
