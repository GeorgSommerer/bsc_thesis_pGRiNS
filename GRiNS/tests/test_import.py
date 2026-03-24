import pytest

def test_import_package():
    try:
        import grins
    except ImportError as e:
        pytest.fail(f"Failed to import package: {e}")
