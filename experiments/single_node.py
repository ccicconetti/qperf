"""Example SquidASM program to test the reliability of a quantum node

The simulation consists of a single quantum node that repeatedly prepares 
three qubits in an initial random state, then check that the output state
is always the same across consecutive repetitions.

With an ideal quantum node the expected behavior is that no error
ever occur, while errors may happen with a non-ideal quantum node.

Environment variables:

- LOG_LEVEL: controls the logging level
- NUM_ITERATIONS: specifies the number of tests to perform
- DEVICE_TYPE: quantum processor type, can be one of: GENERIC, NV
"""

import logging
import numpy as np
from numpy import linalg as LA

from netqasm.sdk.qubit import Qubit
from netqasm.sdk.toolbox.state_prep import set_qubit_state

from squidasm.util import get_qubit_state
from squidasm.run.stack.config import (
    NVQDeviceConfig,
    GenericQDeviceConfig,
    StackConfig,
    StackNetworkConfig,
)
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

from utils import getenv_or_default


class ClientProgram(Program):
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="client_program",
            csockets=[],
            epr_sockets=[],
            max_qubits=3,
        )

    def __init__(self):
        self._logger = LogManager.get_stack_logger(self.__class__.__name__)
        self._last_state = None
        self._phis = [np.random.random() * np.pi for _ in range(3)]
        self._thetas = [np.random.random() * np.pi for _ in range(3)]
        self._norms = []

    def run(self, context: ProgramContext):
        conn = context.connection

        q1 = Qubit(conn)
        q2 = Qubit(conn)
        q3 = Qubit(conn)
        for elem in zip([q1, q2, q3], self._phis, self._thetas):
            set_qubit_state(elem[0], elem[1], elem[2])

        q1.cnot(q2)
        q2.cnot(q3)
        q3.cnot(q1)

        yield from conn.flush()

        new_state = get_qubit_state(q1, "client", True)
        if self._last_state is not None:
            self._norms.append(LA.norm(new_state - self._last_state))
            if self._norms[-1] > 0:
                self._last_state = None
                return {}

        self._last_state = new_state

        return {}


if __name__ == "__main__":
    log_level = logging.getLevelName(getenv_or_default("LOG_LEVEL", "WARNING"))

    device_type = getenv_or_default("DEVICE_TYPE", "NV")
    assert device_type in ["NV", "GENERIC"]

    qdevice_cfg = (
        NVQDeviceConfig.perfect_config()
        if device_type == "NV"
        else GenericQDeviceConfig.perfect_config()
    )
    qdevice_cfg.num_qubits = 3
    client = StackConfig(
        name="client",
        qdevice_typ="nv" if device_type == "NV" else "generic",
        qdevice_cfg=qdevice_cfg,
    )
    cfg = StackNetworkConfig(stacks=[client], links=[])

    num_times = int(getenv_or_default("NUM_ITERATIONS", "10"))

    client_program = ClientProgram()
    client_program._logger.setLevel(log_level)

    run(cfg, {"client": client_program}, num_times=num_times)

    print(f"{np.count_nonzero(client_program._norms) / len(client_program._norms)}")
