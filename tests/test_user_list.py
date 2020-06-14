import tempfile

import pytest


@pytest.fixture
def test_user_exists(client):
    rv = client.get('/get/users')
    assert len(rv) > 1
