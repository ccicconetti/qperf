"""A quantum link performance measurement tool for SquidASM

Simulation that performs a swap test between qubits teleported from
a node to another across a two-node quantum network with depolarizing
quantum link.

The swap test provides information on the end-to-end fidelity of
all the operations involved in the teleportation: EPR pair exchange
and local quantum operations performed by the receiver.

The simulation parameters and output can be controlled through
command-line options, use --help to check all the available configurations.
"""

import logging
import argparse
from os import getenv
import numpy as np

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

from utils import fredkin


def compute_s(outer_list: list) -> float:
    """Compute the swap test output from a number of samples

    Parameters
    ----------
    outer_list: list
        List of list of binary values

    Returns
    -------
    float
        The different between 1 and twice the expectation of the values passed.
    """
    sum = 0.0
    count = 0
    for inner_list in outer_list:
        count += len(inner_list)
        for value in inner_list:
            sum += value

    assert count > 0

    return 1.0 - 2.0 * sum / count


def create_network(link_noise: float):
    """Create a simple network with two NV quantum nodes interconnected
    by a depolarising link with configurable link noise.

    Parameters
    ----------
    link_noise: float
        The noise of the link between the two nodes, where 0 means ideal
        link (no depolarization).

    Returns
    -------
    StackNetworkConfig
        the network created
    """
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
        """Initialize the sender program

        Parameters
        ----------
        P: int
            The number of measurements to make.
        phi: float
            The phi angle to rotate the initial state.
        theta: float
            The theta angle to rotate the initial state.
        LOG_LEVEL: str
            A string representation of the logging level to enable.
        """
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
            max_qubits=3,
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
            max_qubits=3,
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
                    q1 = Qubit(connection)
                    q1.H()
                    fredkin(q1, q2, epr)
                    q1.H()
                    m = q1.measure()
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
    parser = argparse.ArgumentParser(
        "Measure the performance of a quantum link through swap tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Name of the file where to save the output samples",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Show the samples on a plot"
    )
    parser.add_argument(
        "--log-level", type=str, default="WARNING", help="Set the log level"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed to use for the initialization of the pseudo-random number generators",
    )
    parser.add_argument(
        "--num-experiments", type=int, default=5, help="Number of experiments to run"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples per experiment"
    )
    args = parser.parse_args()

    x_values = np.arange(0, 1, 0.1)
    s_values = np.zeros(len(x_values))
    e_values = np.zeros(len(x_values))
    for cnt, x in zip(range(len(x_values)), x_values):
        samples = np.zeros(args.num_experiments)
        for experiment_id in range(args.num_experiments):
            # Create a network configuration
            cfg = create_network(x)

            # Draw random values to initialize the states to be teleported
            rng = np.random.default_rng(args.seed)
            (phi, theta) = (
                rng.random() * np.pi,
                rng.random() * np.pi,
            )

            # Create instances of programs to run
            sender_program = SenderProgram(args.num_samples, phi, theta, args.log_level)
            receiver_program = ReceiverProgram(args.log_level)

            # Run the simulation.
            swap_measurements, _ = run(
                config=cfg,
                programs={"Receiver": receiver_program, "Sender": sender_program},
                num_times=1,
            )

            samples[experiment_id] = compute_s(swap_measurements)

        s_values[cnt] = np.mean(samples)
        e_values[cnt] = np.std(samples)

    if args.output != "":
        with open(args.output, "w") as outfile:
            for x, s, e in zip(x_values, s_values, e_values):
                outfile.write(f"{x:4f} {s:4f} {e:4f}\n")

    if args.plot:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(x_values, s_values, "b-", linewidth=2)
        ax.fill_between(
            x_values, s_values - e_values, s_values + e_values, linewidth=0, alpha=0.2
        )

        ax.set(xlabel="link noise rate", ylabel="fidelity")
        ax.grid()

        plt.show()
