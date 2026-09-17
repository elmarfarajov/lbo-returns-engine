import pytest

from lbolab.examples import apex_rollup, helios_carveout
from lbolab.model import run_lbo
from lbolab.returns import equity_returns


@pytest.fixture(scope="session")
def helios():
    return helios_carveout()


@pytest.fixture(scope="session")
def apex():
    return apex_rollup()


@pytest.fixture(scope="session")
def helios_result(helios):
    return run_lbo(helios)


@pytest.fixture(scope="session")
def apex_result(apex):
    return run_lbo(apex)


@pytest.fixture(scope="session")
def helios_returns(helios_result):
    return equity_returns(helios_result)
