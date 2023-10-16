from os import getenv
import numpy as np

from netqasm.sdk.qubit import Qubit
from netqasm.sdk.toolbox.gates import toffoli_gate
from netqasm.sdk.toolbox.state_prep import set_qubit_state

from netsquid.qubits.dmutil import dm_fidelity

from squidasm.util import get_qubit_state, get_reference_state


def fredkin(control: Qubit, input1: Qubit, input2: Qubit) -> None:
    """Perform a Fredkin gate with `control` as control qubit and
    and `input1` and `input2` as target to be swapped.

    See https://en.wikipedia.org/wiki/Fredkin_gate

    Parameters
    ----------
    control: Qubit
        The control qubit.
    input1: Qubit
        The first input qubit.
    input2: Qubit
        The second input qubit.
    """

    input2.cnot(input1)
    toffoli_gate(control, input1, input2)
    input2.cnot(input1)


def getenv_or_default(env_var: str, default_value: str) -> str:
    """Return the value of an environment variable or a default value.

    Parameters
    ----------
    env_var: str
        The name of the environment variable to be queried.
    default_value: str
        The value returned if env_var is not defined.

    Returns
    -------
    str
        The environment variable or the default value.
    """
    value = getenv(env_var)
    if value is None:
        return default_value
    assert isinstance(value, str)
    return value


def random_init_qubits(qubits: list) -> list:
    """Initialize qubits to random values through a rotation according
    to phi/theta values.

    Parameters
    ----------
    qubits: list
        The qubits to be initialized.

    Returns
    -------
    list
        The list of phi/theta values used for the random rotations
        for each qubit.
    """
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
    """Compute the fidelity of a qubit with respect to a reference one.

    Parameters
    ----------
    program_name: str
        The name of the program to access the state of the qubit.
    qubit: Qubit
        The qubit for which the fidelity is computed.
    phi: float
        The phi rotation of the reference qubit.
    theta: float
        The theta rotation of the reference qubit.

    Returns
    -------
    float
        The fidelity computed, in [0,1].
    """
    return dm_fidelity(
        get_qubit_state(qubit, program_name),
        get_reference_state(phi, theta),
        squared=False,
        dm_check=False,
    )
