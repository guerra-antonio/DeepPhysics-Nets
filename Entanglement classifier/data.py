import numpy as np

import torch

from tqdm import tqdm
from useful_functions import werner_state, partial_transpose, random_state

import numpy as np

def data_peres(rank = None):
    # Generates a random quantum state and applies the Peres-Horodecki (PPT) criterion to check for entanglement.
    
    # Parameters:
    # rank (int, optional): Rank of the random quantum state. If None, a random rank is used.
    
    # Returns:
    # tuple: The generated state (A) and a label (1 if entangled, 0 otherwise).
    
    # Generate a random quantum state (density matrix)
    A = random_state(rank = rank)
    
    # Compute the partial transpose on subsystem A (dimension 2x2)
    ptA = partial_transpose(A, (2, 2), 0)
    
    # Compute the eigenvalues of the partial transpose and take only the real part
    eigsA = np.real(np.linalg.eig(ptA)[0])

    # Check for negative eigenvalues to determine if the state is entangled
    if np.any(eigsA < 0):
        # If any eigenvalue is negative, the state is entangled (label = 1)
        return A, 1
    else:
        # If no eigenvalue is negative, the state is separable (label = 0)
        return A, 0

def create_dataset(num_samples = int, criterion = str, rank = None):
    # Creates a dataset of quantum states and labels based on a given criterion.
    
    # Parameters:
    # num_samples (int): Number of samples to generate.
    # criterion (str): Criterion for generating the dataset, either 'werner' or 'ppt'.
    
    # Returns:
    # X (torch.Tensor): Tensor of shape (num_samples, 2, 4, 4) containing the real and imaginary parts of the states.
    # Y (torch.Tensor): Tensor of shape (num_samples, 1) containing the labels for each state.
    
    X = torch.Tensor()
    Y = torch.Tensor()
    
    if criterion == "werner":
        for _ in tqdm(range(num_samples), desc="Generating Werner states"):
            # Generate a random Werner state and its parameter p
            w_state, p = werner_state()

            # Label: 1 if p > 1/3 (entangled), else 0
            label = 1 if p > 1/3 else 0
            label = torch.Tensor([label])
            
            # Separate real and imaginary parts of the Werner state
            w_state = np.stack((np.real(w_state), np.imag(w_state)))
            w_state = torch.Tensor(w_state)

            # Append the state and label to torch.tensors
            X = torch.cat((X, w_state), dim = 0)
            Y = torch.cat((Y, label))
    
    elif criterion == "ppt":
        for _ in tqdm(range(num_samples), desc="Generating states with Peres criterion"):
            # Generate a random state and its corresponding Peres criterion label
            state, label = data_peres(rank = None)

            label = torch.Tensor([label])
            state = np.stack((np.real(state), np.imag(state)))
            state = torch.Tensor(state)

            # Append the state and label to torch.tensors
            X = torch.cat((X, state), dim = 0)
            Y = torch.cat((Y, label), dim = 0)
    
    else:
        raise ValueError("Please specify 'werner' or 'ppt' as criterion.")
    
    # Ensure the correct shapes: (num_samples, 2, 4, 4) for X and (num_samples, 1) for Y
    X = X.view(num_samples, 2, 4, 4)
    Y = Y.view(num_samples, 1)
    
    return X, Y