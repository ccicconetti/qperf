import math
import random
import os
import logging
import time
import argparse
import pandas as pd

from statistics import mean

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
)
from qiskit.circuit import Qubit, Clbit

from noise_model_wrapper import NoiseModelWrapper
from utils import to_sec

logger = logging.getLogger(__name__)


def bell_pair(qc: QuantumCircuit, q1: Qubit, q2: Qubit):
    qc.h(q1)
    qc.cx(q1, q2)


def teleport(
    qc: QuantumCircuit, q1: Qubit, q2: Qubit, q3: Qubit, crz: Clbit, crx: Clbit
):
    qc.cx(q1, q2)
    qc.h(q1)

    qc.measure(q1, crz)
    qc.measure(q2, crx)

    qc.z(q3).c_if(crz, 1)
    qc.x(q3).c_if(crx, 1)


def make_circuit(num_filler_gates: int):
    """Create a circuit that performs teleportation of two qubits and a swap test

    Parameters
    ----------
    num_filler_gates: int
        The number of filler gates to be added before the swap test
    """

    # Initialize circuit and registers
    psi1 = QuantumRegister(1, "psi1")
    phi1 = QuantumRegister(2, "phi1")
    c1 = ClassicalRegister(2, "c1")
    psi2 = QuantumRegister(1, "psi2")
    phi2 = QuantumRegister(2, "phi2")
    c2 = ClassicalRegister(2, "c2")
    ancilla = QuantumRegister(1, "ancilla")
    outfiller = ClassicalRegister(2, "outfiller")
    filler = QuantumRegister(2, "filler")
    out = ClassicalRegister(1, "out")
    qc = QuantumCircuit(psi1, phi1, c1, psi2, phi2, c2, ancilla, filler, outfiller, out)

    # Initialize |psi> qubits to be teleported to the same random state
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)
    logger.debug("theta = {}, phi = {}".format(theta, phi))
    for qbit in [psi1, psi2]:
        qc.rz(phi, qbit)
        qc.rx(theta, qbit)

    # Create Bell pairs |Phi>
    bell_pair(qc, phi1[0], phi1[1])
    bell_pair(qc, phi2[0], phi2[1])
    qc.barrier()

    # Teleport |psi> to |Phi>[1]
    teleport(qc, psi1, phi1[0], phi1[1], c1[0], c1[1])
    teleport(qc, psi2, phi2[0], phi2[1], c2[0], c2[1])
    qc.barrier()

    # Add delay
    # note: cannot use qc.delay() as this cannot be transpiled on a real QC
    if num_filler_gates > 0:
        qc.h(filler[0])
        qc.h(filler[1])
        for i in range(num_filler_gates):
            if i % 4 == 0:
                qc.cx(filler[0], filler[1])
            elif i % 4 == 1:
                qc.rz(random.uniform(0, 2 * math.pi), filler[0])
            elif i % 4 == 2:
                qc.cx(filler[1], filler[0])
            else:
                assert i % 4 == 3
                qc.rz(random.uniform(0, 2 * math.pi), filler[1])

        qc.measure(filler, outfiller)
        qc.barrier()

    # Implement swap test
    qc.h(ancilla)
    qc.cswap(ancilla, phi1[1], phi2[1])
    qc.h(ancilla)
    qc.measure(ancilla, out)

    logger.debug("\n{}".format(qc.draw(output="text")))

    return qc


def analyze_results(result, shots: int) -> float:
    counter = 0.0
    total = 0.0
    for k, v in result.get_counts().items():
        logger.debug(f"{k}: {v}")
        total += v
        if k[0] == "1":
            counter += v
    assert total == shots

    estimated_fidelity = 1.0 - (2.0 * counter) / total
    logger.info("estimated fidelity: {}".format(estimated_fidelity))
    return estimated_fidelity


def main():
    parser = argparse.ArgumentParser(
        "Simulate qperf measurements on a QC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend", type=str, default="ideal", help="Name of the backend to simulate"
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
        "--num-samples", type=int, default=1024, help="Number of samples per experiment"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Delay to be added before swap test, in us",
    )
    args = parser.parse_args()

    # Initialize pseudo-random number generator seed
    random.seed(args.seed)

    # Configure logging
    log_level = logging.getLevelName(args.log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logging.getLogger(__name__).setLevel(log_level)
    logger.addHandler(ch)
    logging.getLogger("noise_model_wrapper").setLevel(log_level)
    logging.getLogger("noise_model_wrapper").addHandler(ch)

    # Load backend
    noise_model_wrapper = NoiseModelWrapper(backend_name=args.backend, no_save=False)

    # Check backend properties
    properties = noise_model_wrapper.backend_properties()
    gate_durations = []
    if properties is not None:
        for gate in properties.gates:
            if str(gate.gate).lower() == "x":
                for param in gate.parameters:
                    if param.name == "gate_length":
                        gate_durations.append(to_sec(param.value, param.unit))
    avg_gate_duration = mean(gate_durations)
    num_filler_gates = int(round(1e-6 * args.delay / avg_gate_duration))
    logger.info(
        f"average gate duration: {avg_gate_duration} s ({num_filler_gates} filler gates)"
    )

    # Execute experiment
    ts_start = time.time()
    data = []
    for ndx_experiment in range(args.num_experiments):
        # Create circuit
        qc = make_circuit(num_filler_gates=num_filler_gates)

        # Run simulation
        result = noise_model_wrapper.execute(qc, shots=args.num_samples, seed=args.seed)

        # Analyze results
        data.append(
            (
                args.backend,
                args.seed,
                ndx_experiment,
                time.time(),
                analyze_results(result=result, shots=args.num_samples),
            )
        )

    duration = time.time() - ts_start
    logger.info("experiment duration: {}".format(duration))

    # Save experiment data
    if args.output:
        df_new = pd.DataFrame(
            data, columns=["backend", "seed", "ndx_experiment", "ts", "ratio"]
        )

        if os.path.exists(args.output):
            df_old = pd.read_csv(args.output)
            df = pd.concat([df_old, df_new])

        else:
            df = df_new

        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
