import numpy as np

from netqasm.sdk.qubit import Qubit
from netqasm.sdk.toolbox.state_prep import set_qubit_state

from netsquid.qubits.dmutil import dm_fidelity

from squidasm.util import get_qubit_state, get_reference_state
from squidasm.run.stack.config import NVQDeviceConfig, StackConfig, StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

from utils import getenv_or_default, fredkin


def to_degrees(angle_rad: float) -> int:
    return int(round(angle_rad * 180 / np.pi))


def fidelity(qubit: Qubit, phi: float, theta: float) -> float:
    return dm_fidelity(
        get_qubit_state(qubit, "client"),
        get_reference_state(phi, theta),
        squared=False,
        dm_check=False,
    )


class ClientProgram(Program):
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="client_program",
            csockets=[],
            epr_sockets=[],
            max_qubits=3,
        )

    def run(self, context: ProgramContext):
        conn = context.connection
        logger = LogManager.get_stack_logger(self.__class__.__name__)

        (phi1, theta1) = (
            np.random.random() * np.pi / 2,
            np.random.random() * np.pi / 2,
        )
        (phi2, theta2) = (phi1 + np.pi / 3, theta1 + np.pi / 3)
        logger.info(
            f"qubit1: phi {to_degrees(phi1)} degrees, theta {to_degrees(theta1)} degrees"
        )
        logger.info(
            f"qubit2: phi {to_degrees(phi2)} degrees, theta {to_degrees(theta2)} degrees"
        )

        ctrl = Qubit(conn)
        ctrl.X()
        qubit1 = Qubit(conn)
        set_qubit_state(qubit1, phi1, theta1)
        qubit2 = Qubit(conn)
        set_qubit_state(qubit2, phi2, theta2)

        fredkin(ctrl, qubit1, qubit2)
        yield from conn.flush()

        (fid_1_1, fid_1_2, fid_2_1, fid_2_2) = (
            fidelity(qubit1, phi1, theta1),
            fidelity(qubit1, phi2, theta2),
            fidelity(qubit2, phi1, theta1),
            fidelity(qubit2, phi2, theta2),
        )

        logger.info(
            f"fidelities {fid_1_1:.2f} {fid_1_2:.2f} {fid_2_1:.2f} {fid_2_2:.2f}"
        )

        return {}


if __name__ == "__main__":
    LogManager.set_log_level(getenv_or_default("LOG_LEVEL", "WARNING"))

    qdevice_cfg = NVQDeviceConfig.perfect_config()
    qdevice_cfg.num_qubits = 3
    client = StackConfig(
        name="client",
        qdevice_typ="nv",
        qdevice_cfg=qdevice_cfg,
    )
    cfg = StackNetworkConfig(stacks=[client], links=[])

    run(cfg, {"client": ClientProgram()}, num_times=10)
