import pytest

from chan_signal_studio import analyze
from chan_signal_studio.data import generate_bars


@pytest.fixture(scope="session")
def analyzed_series():
    return [analyze(generate_bars(2000, seed=s)) for s in range(1, 21)]
