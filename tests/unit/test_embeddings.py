"""Unit tests for src/embeddings.py — Vertex AI embedder."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.embeddings import GoogleEmbedder, _DEFAULT_DIM, _DEFAULT_MODEL


def _make_embedding(values: list[float]) -> MagicMock:
    e = MagicMock()
    e.values = values
    return e


def _fake_client(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.embeddings = [_make_embedding(v) for v in vectors]
    client = MagicMock()
    client.models.embed_content.return_value = response
    return client


@patch("src.embeddings.genai.Client")
def test_init_uses_vertex_ai(mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    GoogleEmbedder(project="my-project", location="us-east1")
    mock_client_cls.assert_called_once_with(
        vertexai=True,
        project="my-project",
        location="us-east1",
    )


@patch("src.embeddings.genai.Client")
def test_init_reads_env_vars(mock_client_cls, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")
    mock_client_cls.return_value = MagicMock()
    GoogleEmbedder()
    mock_client_cls.assert_called_once_with(
        vertexai=True,
        project="env-project",
        location="us-central1",
    )


@patch("src.embeddings.genai.Client")
def test_init_no_api_key_required(mock_client_cls):
    """No GEMINI_API_KEY env var should not raise."""
    mock_client_cls.return_value = MagicMock()
    embedder = GoogleEmbedder(project="p", location="l")
    assert embedder.model == _DEFAULT_MODEL
    assert embedder.output_dimensionality == _DEFAULT_DIM


@patch("src.embeddings.genai.Client")
def test_embed_documents_calls_retrieval_document(mock_client_cls):
    client = _fake_client([[0.1] * 768, [0.2] * 768])
    mock_client_cls.return_value = client

    embedder = GoogleEmbedder(project="p", location="l")
    result = embedder.embed_documents(["doc one", "doc two"])

    assert result.shape == (2, 768)
    call_kwargs = client.models.embed_content.call_args
    assert call_kwargs.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


@patch("src.embeddings.genai.Client")
def test_embed_query_calls_retrieval_query(mock_client_cls):
    client = _fake_client([[0.5] * 768])
    mock_client_cls.return_value = client

    embedder = GoogleEmbedder(project="p", location="l")
    result = embedder.embed_query("hello")

    assert result.shape == (768,)
    call_kwargs = client.models.embed_content.call_args
    assert call_kwargs.kwargs["config"].task_type == "RETRIEVAL_QUERY"


@patch("src.embeddings.genai.Client")
def test_embed_documents_empty_input(mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    embedder = GoogleEmbedder(project="p", location="l")
    result = embedder.embed_documents([])
    assert result.shape == (0, 768)
    mock_client_cls.return_value.models.embed_content.assert_not_called()


@patch("src.embeddings.genai.Client")
def test_l2_normalisation(mock_client_cls):
    raw = [3.0, 4.0] + [0.0] * 766
    client = _fake_client([raw])
    mock_client_cls.return_value = client

    embedder = GoogleEmbedder(project="p", location="l", output_dimensionality=768)
    result = embedder.embed_query("x")

    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-5
