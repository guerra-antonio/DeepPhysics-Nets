import numpy as np
import torch
import random

from useful_functions import random_state, state_evol

# For train the model we require data, for simplicity we define a function for randomly generates the data

def generates_data(N_data = int, n_qubits = 2, rank = None, Hamiltonian = None):
    # Generates a dataset of quantum states evolved over time.

    # Parameters:
    # N_data (int): Number of data points to generate.
    # n_qubits (int): Number of qubits in the system.
    # rank (int, optional): Rank of the random quantum state. If None, it will be randomly selected.
    # H (np.ndarray, optional): Hamiltonian matrix for the system. If None, the identity matrix is used.

    # Returns:
    # tuple: A tuple containing two lists:
    #     - input_data: List of tuples (initial_state_tensor, time) for each data point.
    #     - output_data: List of final_state_tensor for each data point.

    input_data  = [], []   # To store initial states and time
    output_data = []       # To store final states

    # If no Hamiltonian is provided, use the identity matrix as the Hamiltonian
    if Hamiltonian is None:
        Hamiltonian = np.eye(2**n_qubits)

    for n in range(N_data):
        # Generate a random initial state with the given number of qubits and rank
        rho_in = random_state(n_qubits = n_qubits, rank = rank)
        real_in = np.real(rho_in).astype(np.float32)
        imag_in = np.imag(rho_in).astype(np.float32)

        # Stack real and imaginary parts into a tensor
        nn_in = torch.tensor(np.stack((real_in, imag_in)))

        # Generate a random time for the evolution
        time = torch.tensor([random.random()])

        # Evolve the initial state over time using the Hamiltonian
        rho_out = state_evol(rho_initial = rho_in, time = time, Hamiltonian = Hamiltonian)
        real_out = np.real(rho_out).astype(np.float32)
        imag_out = np.imag(rho_out).astype(np.float32)

        # Stack real and imaginary parts of the evolved state into a tensor
        nn_out = torch.tensor(np.stack((real_out, imag_out)))

        # Append the input data (initial state and time) and the output data (final state)
        input_data[0].append(nn_in)  # Initial state tensor
        input_data[1].append(time)   # Time of evolution
        output_data.append(nn_out)   # Final state tensor

    return input_data, output_data