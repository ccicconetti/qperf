import logging
from os import getenv
import numpy as np

import matplotlib.pyplot as plt

from netsquid.qubits.dmutil import dm_fidelity
from netsquid.qubits import ketstates, ketutil
from netsquid.util.simtools import sim_time, MILLISECOND

from netqasm.sdk import Qubit
from netqasm.sdk.toolbox import set_qubit_state
from netqasm.sdk.classical_communication.message import StructuredMessage

from squidasm.run.stack.config import (
    NVQDeviceConfig,
    StackConfig,
    DepolariseLinkConfig,
    LinkConfig,
    StackNetworkConfig,
)
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state

from utils import getenv_or_default, fredkin


def compute_s(outer_list: list) -> float:
    sum = 0.0
    count = 0
    for inner_list in outer_list:
        count += len(inner_list)
        for value in inner_list:
            sum += value

    assert count > 0

    return 1.0 - 2.0 * sum / count


def create_network(link_noise: float):
    node_names = ["Receiver", "Sender"]

    qdevice_cfg = NVQDeviceConfig.perfect_config()
    qdevice_cfg.num_qubits = 10
    stacks = [
        StackConfig(name=name, qdevice_typ="nv", qdevice_cfg=qdevice_cfg)
        for name in node_names
    ]

    link_cfg = DepolariseLinkConfig(
        fidelity=1 - link_noise * 3 / 4, t_cycle=1000, prob_success=1
    )
    link = LinkConfig(
        stack1=node_names[0], stack2=node_names[1], typ="depolarise", cfg=link_cfg
    )
    return StackNetworkConfig(stacks=stacks, links=[link])


class SenderProgram(Program):
    PEER_NAME = "Receiver"

    def __init__(self, P: int, phi: float, theta: float, LOG_LEVEL: str):
        self._P = P
        self._phi = phi
        self._theta = theta
        self._logger = LogManager.get_stack_logger(self.__class__.__name__)
        self._logger.setLevel(logging.getLevelName(LOG_LEVEL))
        self._logger.info(f"phi = {phi}, theta = {theta}")

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="qperf",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=10,
        )

    def run(self, context: ProgramContext):
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        # Send the number of tests to the peer
        csocket.send_structured(StructuredMessage("Config", self._P))
        for test in range(self._P):
            for stage in range(2):
                self._logger.debug(f"test#{test}, stage#{stage}")

                # Wait from the peer before proceeding
                msg = yield from csocket.recv_structured()
                assert msg.header == "CTG"
                assert int(msg.payload) == stage

                # Create the a qubit to teleport
                q = Qubit(connection)
                set_qubit_state(q, self._phi, self._theta)
                yield from connection.flush()

                self._logger.debug(
                    f"State to be teleported:\n{get_qubit_state(q, 'Sender')}"
                )

                # Create an EPR pair
                epr = epr_socket.create_keep()[0]

                # Teleport the local qubit to the remote node
                q.cnot(epr)
                q.H()
                (m1, m2) = (q.measure(), epr.measure())

                # Run the circuit
                yield from connection.flush()

                # Send the measurements to the peer
                csocket.send_structured(
                    StructuredMessage("Measurements", f"{test},{stage},{m1},{m2}")
                )

        self._logger.debug(f"Finished, duration = {sim_time(MILLISECOND)} ms")

        return {}


class ReceiverProgram(Program):
    PEER_NAME = "Sender"

    def __init__(self, LOG_LEVEL: str):
        self._logger = LogManager.get_stack_logger(self.__class__.__name__)
        self._logger.setLevel(logging.getLevelName(LOG_LEVEL))

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="qperf",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=4,
        )

    def run(self, context: ProgramContext):
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        # Receive from peer the configuration
        msg = yield from csocket.recv_structured()
        assert msg.header == "Config"
        P = int(msg.payload)
        self._logger.debug(f"Received configuration from peer: {P}")

        swap_measurements = []

        for test in range(P):
            # Local qubits
            q1 = Qubit(connection)
            q2 = Qubit(connection)
            q3 = Qubit(connection)
            yield from connection.flush()

            for stage in range(2):
                self._logger.debug(f"test#{test}, stage#{stage}")

                # Send a clear-to-go to the sender
                csocket.send_structured(StructuredMessage("CTG", str(stage)))

                # Receive the EPR pairs
                epr = epr_socket.recv_keep()[0]
                yield from connection.flush()

                self._logger.debug(
                    f"Bell state received:\n{get_qubit_state(epr, 'Receiver')}"
                )

                # Receive the corrections
                msg = yield from csocket.recv_structured()
                assert msg.header == "Measurements"
                recv_data = [int(x) for x in msg.payload.split(",")]
                assert len(recv_data) == 4
                (test_r, stage_r, m1, m2) = recv_data
                assert test == test_r
                assert stage == stage_r
                self._logger.debug(f"Measurements received: {m1}, {m2}")

                # Perform the X,Z corrections
                if m1 == 1:
                    epr.Z()
                if m2 == 1:
                    epr.X()
                yield from connection.flush()

                self._logger.debug(
                    f"Teleported state:\n{get_qubit_state(epr, 'Receiver')}"
                )

                if stage == 0:
                    # Create a local EPR pair.
                    q1.H()
                    q1.cnot(q2)
                    yield from connection.flush()

                    # Teleport the received qubit to another local qubit
                    epr.cnot(q1)
                    epr.H()
                    m1_local = epr.measure()
                    m2_local = q1.measure()
                    yield from connection.flush()

                    if int(m1_local) == 1:
                        q2.Z()
                    if int(m2_local) == 1:
                        q2.X()
                    yield from connection.flush()

                    self._logger.debug(
                        f"Local state:\n{get_qubit_state(q2, 'Receiver')}"
                    )

                else:
                    assert stage == 1

                    q3.H()
                    fredkin(q3, q2, epr)
                    q3.H()
                    m = q3.measure()
                    yield from connection.flush()

                    swap_measurements.append(int(m))

                    self._logger.debug(f"Swap test measurement: {int(m)}")

                    # Measure qubits to reset their state
                    q2.measure()
                    epr.measure()
                    yield from connection.flush()

        self._logger.debug(f"Finished, duration = {sim_time(MILLISECOND)} ms")

        return swap_measurements


if __name__ == "__main__":
    x_values = []
    s_values = []
    for link_noise_int in range(0, 81, 20):
        link_noise = link_noise_int / 100.0

        x_values.append(link_noise)

        # Create a network configuration
        cfg = create_network(link_noise)

        # Read the number of tests to perform from environment variable P
        OUTPUT = getenv_or_default("OUTPUT", "")
        P = int(getenv_or_default("P", "10"))
        LOG_LEVEL = getenv_or_default("LOG_LEVEL", "WARNING")
        SEED = int(getenv_or_default("SEED", "0"))
        PLOT = getenv_or_default("PLOT", "")

        # Draw random values to initialize the states to be teleported
        rng = np.random.default_rng(SEED)
        (phi, theta) = (
            rng.random() * np.pi,
            rng.random() * np.pi,
        )

        # Create instances of programs to run
        sender_program = SenderProgram(P, phi, theta, LOG_LEVEL)
        receiver_program = ReceiverProgram(LOG_LEVEL)

        # Run the simulation.
        swap_measurements, _ = run(
            config=cfg,
            programs={"Receiver": receiver_program, "Sender": sender_program},
            num_times=1,
        )

        s_values.append(compute_s(swap_measurements))

    assert len(s_values) == len(x_values)

    if OUTPUT != "":
        with open(OUTPUT, "w") as outfile:
            for x, s in zip(x_values, s_values):
                outfile.write(f"{x} {s}\n")

    if PLOT != "":
        fig, ax = plt.subplots()
        ax.plot(x_values, s_values)

        ax.set(xlabel="link noise rate", ylabel="fidelity")
        ax.grid()

        plt.show()
