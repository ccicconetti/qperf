"""Example SquidASM program to test the Fredkin gate

The simulation consists of a single NV quantum node that repeatedly prepares 
pairs of qubits in an initial random rate and then performs a
swap test using a Fredkin gate, also known as controlled SWAP gate.

Environment variables:

- LOG_LEVEL: controls the logging level
- NUM_ITERATIONS: specifies the number of tests to perform
"""

import logging
import numpy as np

from netqasm.sdk.qubit import Qubit

from squidasm.run.stack.config import NVQDeviceConfig, StackConfig, StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

from utils import getenv_or_default, fredkin, random_init_qubits, fidelity


def to_degrees(angle_rad: float) -> int:
    return int(round(angle_rad * 180 / np.pi))


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
        self._errors: int = 0

    def run(self, context: ProgramContext):
        conn = context.connection

        flip = np.random.random() > 0.5

        ctrl = Qubit(conn)
        if flip:
            ctrl.X()
        qubit1 = Qubit(conn)
        qubit2 = Qubit(conn)
        random_values = random_init_qubits([qubit1, qubit2])
        assert len(random_values)
        phi1 = random_values[0][0]
        phi2 = random_values[1][0]
        theta1 = random_values[1][1]
        theta2 = random_values[1][1]

        self._logger.info(
            f"qubit1: phi {to_degrees(phi1)} degrees, theta {to_degrees(theta1)} degrees"
        )
        self._logger.info(
            f"qubit2: phi {to_degrees(phi2)} degrees, theta {to_degrees(theta2)} degrees"
        )

        fredkin(ctrl, qubit1, qubit2)
        yield from conn.flush()

        (fid_1_1, fid_1_2, fid_2_1, fid_2_2) = (
            fidelity("client", qubit1, phi1, theta1),
            fidelity("client", qubit1, phi2, theta2),
            fidelity("client", qubit2, phi1, theta1),
            fidelity("client", qubit2, phi2, theta2),
        )

        self._logger.info(
            f"flip {flip} fidelities {fid_1_1:.2f} {fid_1_2:.2f} {fid_2_1:.2f} {fid_2_2:.2f}"
        )

        fid_prod_higher = fid_1_1 * fid_2_2 if not flip else fid_1_2 * fid_2_1
        fid_prod_lower = fid_1_2 * fid_2_1 if not flip else fid_1_1 * fid_2_2

        if fid_prod_higher <= fid_prod_lower:
            self._errors += 1

        return {}


if __name__ == "__main__":
    log_level = logging.getLevelName(getenv_or_default("LOG_LEVEL", "WARNING"))

    qdevice_cfg = NVQDeviceConfig.perfect_config()
    qdevice_cfg.num_qubits = 3
    client = StackConfig(
        name="client",
        qdevice_typ="nv",
        qdevice_cfg=qdevice_cfg,
    )
    cfg = StackNetworkConfig(stacks=[client], links=[])

    num_times = int(getenv_or_default("NUM_ITERATIONS", "100"))
    client_program = ClientProgram()
    client_program._logger.setLevel(log_level)

    run(cfg, {"client": client_program}, num_times=num_times)

    print(f"error ratio: {client_program._errors / num_times}")
