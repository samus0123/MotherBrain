import os

# Keep TensorFlow's C++ logging out of the test output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: builds a real Keras model (needs TensorFlow)"
    )
