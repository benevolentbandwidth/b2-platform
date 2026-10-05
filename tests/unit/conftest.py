import pytest

from tools.fake_image_detector import google_clients


@pytest.fixture(autouse=True)
def _fresh_google_clients():
    """Clients are shared per process; tests swap the google modules for stubs,
    so each test starts with none cached."""
    google_clients.reset()
    yield
    google_clients.reset()
