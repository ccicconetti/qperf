from os import getenv

from netqasm.sdk.qubit import Qubit
from netqasm.sdk.toolbox.gates import toffoli_gate


def fredkin(control: Qubit, input1: Qubit, input2: Qubit) -> None:
    """Performs a Fredkin gate with `control` as control qubit and
    and `input1` and `input2` as target to be swapped.

    See https://en.wikipedia.org/wiki/Fredkin_gate
    """

    input2.cnot(input1)
    toffoli_gate(control, input1, input2)
    input2.cnot(input1)


def getenv_or_default(env_var: str, default_value: str) -> str:
    value = getenv(env_var)
    if value is None:
        return default_value
    assert isinstance(value, str)
    return value
