from os import getenv
import numpy as np

from netqasm.sdk.qubit import Qubit
from netqasm.sdk.toolbox.gates import toffoli_gate
from netqasm.sdk.toolbox.state_prep import set_qubit_state

from netsquid.qubits.dmutil import dm_fidelity

from squidasm.util import get_qubit_state, get_reference_state


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


def random_init_qubits(qubits: list) -> list:
    ret = []
    for qubit in qubits:
        (phi, theta) = (
            np.random.random() * np.pi,
            np.random.random() * np.pi,
        )
        ret.append((phi, theta))

        set_qubit_state(qubit, phi, theta)

    return ret


def fidelity(program_name: str, qubit: Qubit, phi: float, theta: float) -> float:
    return dm_fidelity(
        get_qubit_state(qubit, program_name),
        get_reference_state(phi, theta),
        squared=False,
        dm_check=False,
    )
