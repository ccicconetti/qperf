"""Wrapper to cache on disk data to run noise simulations with a model from a real IBM backend"""

from ast import literal_eval
import pickle
from os import path
import logging
import numpy as np
from numpy import pi

from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider
from qiskit_aer import QasmSimulator, AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.models import BackendProperties
from qiskit.qasm3 import dump

logger = logging.getLogger(__name__)


class NoiseModelWrapper:
    "Load noise model from IBMQ real quantum computer"

    def __init__(self, backend_name: str, no_save=False):
        """Load a noise model from either a local file or IBMQ"""

        logger.info("Building circuit with noise from '{}'".format(backend_name))

        # If a file exists called like the backend, then load the model from that.
        # In this case, we also try to load a coupling map from
        # a file with .map extension. If it does not exist, no
        # worries, we just assume it is default (i.e., empty).
        noise_filename = "{}.noise".format(backend_name)
        coupling_map_filename = "{}.map".format(backend_name)
        properties_filename = "{}.properties".format(backend_name)

        if backend_name == "ideal":
            self.noise_model = None
            self.coupling_map = None
            self.properties = None

        elif path.exists(noise_filename):
            logger.info("Loading noise model from {}".format(noise_filename))
            with open(noise_filename, "rb") as infile:
                self.noise_model = NoiseModel.from_dict(pickle.load(infile))

            self.coupling_map = None
            if path.exists(coupling_map_filename):
                logger.info(
                    "Loading coupling map from {}".format(coupling_map_filename)
                )
                with open(coupling_map_filename, "r") as coupling_infile:
                    self.coupling_map = CouplingMap(
                        literal_eval(coupling_infile.read())
                    )

            self.properties = None
            if path.exists(properties_filename):
                logger.info("Loading properties from {}".format(properties_filename))
                with open(properties_filename, "rb") as properties_infile:
                    self.properties = BackendProperties.from_dict(
                        pickle.load(properties_infile)
                    )

        # Otherwise, load the noise model from IBMQ (requires token)
        # account properties to be stored in default location
        # and save the noise model for future use, unless the no_save flag is set
        else:
            # Load IBM credentials and backend
            provider = IBMProvider()
            try:
                backend_real = provider.get_backend(name=backend_name)
            except QiskitBackendNotFoundError:
                raise RuntimeError(
                    "backend {} not found, available backends: {}".format(
                        backend_name, list(provider.backends())
                    ),
                )

            # Generate noise model from real backend
            self.noise_model = NoiseModel.from_backend(backend_real)

            # Get coupling map from backend
            self.coupling_map = backend_real.configuration().coupling_map

            # Get properties from backend
            self.properties = backend_real.properties()

            # Save the model and coupling map (if not specified) to file
            if not no_save:
                logger.info(
                    "Saving to {} the noise model for future use".format(noise_filename)
                )
                with open(noise_filename, "wb") as outfile:
                    pickle.dump(self.noise_model.to_dict(), outfile)

                if self.coupling_map is not None:
                    logger.info(
                        "Saving to {} the coupling map for future use".format(
                            coupling_map_filename
                        )
                    )
                    with open(coupling_map_filename, "w") as coupling_outfile:
                        coupling_outfile.write(str(self.coupling_map))

                if self.properties is not None:
                    logger.info(
                        "Saving to {} the properties for future use".format(
                            properties_filename
                        )
                    )
                    with open(properties_filename, "wb") as properties_outfile:
                        pickle.dump(self.properties.to_dict(), properties_outfile)

    def execute(self, qc, shots: int, seed: int):
        "Execute simulation with noise"

        # Retrieve the backend
        if self.noise_model is None:
            assert self.coupling_map is None
            assert self.properties is None
            backend = QasmSimulator(method="statevector")

        else:
            assert self.coupling_map is not None
            assert self.properties is not None
            backend = AerSimulator(
                noise_model=self.noise_model,
                coupling_map=self.coupling_map,
                basis_gates=self.noise_model.basis_gates,
                method="density_matrix",
                max_parallel_experiments=0,
            )

        # Transpile the circuit for the backend
        transpiled_circuit = transpile(
            qc,
            backend,
            optimization_level=3,
            coupling_map=self.coupling_map,
            basis_gates=self.noise_model.basis_gates,
        )
        logger.debug(f"transpiled circuit:\n{transpiled_circuit}")

        # with open("circuit.qasm", "w") as dumpfile:
        #     dump(transpiled_circuit, dumpfile)

        # Execute the simulation
        result = backend.run(
            transpiled_circuit, shots=shots, seed_simulator=seed
        ).result()

        return result

    def backend_properties(self) -> BackendProperties | None:
        "Return the backend properties"

        return self.properties
