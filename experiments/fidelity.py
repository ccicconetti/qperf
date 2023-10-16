"""Example SquidASM program that measures fidelity of EPR pairs exchanged

Two quantum nodes are interconnected with a quantum link.

The sender initially sends to the receiver the number of EPR pairs that
will be exchanged. Afterwards, they both receive EPR pairs back-to-back,
and the receiver measures the fidelity of the two qubits (this operation
is only possible in simulations because the two states reside in
systems that are physically separated from one another).

The fidelity measurements can be saved in an output file specified by
the environment variable OUTFILE.

The number of EPR pairs to generate are specified by the environment
variable NUM_EPR_PAIRS.

The logging level is set according to the LOG_LEVEL environment variable.
"""

import logging
from os import getenv

from netsquid.qubits.dmutil import dm_fidelity
from netsquid.qubits import ketstates, ketutil

from squidasm.run.stack.config import (
    GenericQDeviceConfig,
    StackConfig,
    DepolariseLinkConfig,
    LinkConfig,
    StackNetworkConfig,
)
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state

from utils import getenv_or_default


def create_network(link_noise: float):
    """Create a simple network with two ideal quantum nodes interconnected
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

    qdevice_cfg = GenericQDeviceConfig.perfect_config()
    qdevice_cfg.num_qubits = 10
    stacks = [
        StackConfig(name=name, qdevice_typ="generic", qdevice_cfg=qdevice_cfg)
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

    def __init__(self, num_epr_pairs: int):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self._num_epr_pairs = num_epr_pairs

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="fidelity",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        # Send the number of expected EPR pairs to the peer
        csocket.send(f"{self._num_epr_pairs}")
        for i in range(self._num_epr_pairs):
            # Create one EPR pair
            epr = epr_socket.create_keep()[0]
            yield from connection.flush()
            self.logger.info(f"#{i} EPR pair created")
            csocket.send("ping")

            # Wait for the peer to perform the fidelity measurement
            msg = yield from csocket.recv()
            assert msg == "pong"

            # Destroy the qubit by measuring it to free the qmemory cell
            epr.measure()
            yield from connection.flush()

        self.logger.info(f"finished")

        return {}


class ReceiverProgram(Program):
    PEER_NAME = "Sender"

    def __init__(self):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="fidelity",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        # Receive from peer the number of expected EPR pairs
        num_epr_pairs = yield from csocket.recv()
        num_epr_pairs = int(num_epr_pairs)
        self.logger.info(f"Receiver expects {num_epr_pairs} EPR pairs")

        fidelities = []
        for i in range(num_epr_pairs):
            epr = epr_socket.recv_keep()[0]
            yield from connection.flush()
            self.logger.info("EPR pair received")

            msg = yield from csocket.recv()
            assert msg == "ping"
            self.logger.info(f"#{i} EPR pair received")
            qubit_sender = get_qubit_state(epr, "Sender")
            qubit_receiver = get_qubit_state(epr, "Receiver")
            f_sender = float(
                dm_fidelity(qubit_sender, ketutil.reduced_dm(ketstates.b00, [0]))
            )
            f_receiver = float(
                dm_fidelity(qubit_receiver, ketutil.reduced_dm(ketstates.b00, [0]))
            )
            assert int(10000 * f_sender) == int(10000 * f_receiver)
            fidelities.append(f_sender)

            # Notify the peer that it can proceed
            csocket.send("pong")

            # Destroy the qubit by measuring it to free the qmemory cell
            epr.measure()
            yield from connection.flush()

        self.logger.info(f"finished")

        return fidelities


if __name__ == "__main__":
    # Create a network configuration
    cfg = create_network(0.95)

    # Create instances of programs to run
    num_epr_pairs = int(getenv_or_default("NUM_EPR_PAIRS", "10"))
    sender_program = SenderProgram(num_epr_pairs)
    receiver_program = ReceiverProgram()

    # Set logging level
    log_level = logging.getLevelName(getenv_or_default("LOG_LEVEL", "WARNING"))
    receiver_program.logger.setLevel(log_level)
    sender_program.logger.setLevel(log_level)

    # Run the simulation.
    fidelities, _ = run(
        config=cfg,
        programs={"Receiver": receiver_program, "Sender": sender_program},
        num_times=1,
    )

    output_filename = getenv_or_default("OUTFILE", "")
    if output_filename:
        with open(output_filename, "w") as outfile:
            for f in fidelities[0]:
                outfile.write(f"{f}\n")
