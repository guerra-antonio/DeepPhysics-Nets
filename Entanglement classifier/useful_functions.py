import numpy as np
import random
from scipy.linalg import expm, sqrtm
import torch

def random_vector(n_qubits = 2):
    # Generates a random unit vector in a Hilbert space of dimension 2**n_qubits.
    
    # Parameters:
    # n_qubits (int): The number of qubits, determines the dimension of the vector (2**n_qubits).
    
    # Returns:
    # np.ndarray: A normalized random complex vector (unit vector).

    dim = 2**n_qubits  # The dimension of the vector space (2^n_qubits)
    
    # Generate a complex random vector
    vector = np.array([2 * np.random.random() - 1 + 1j * (2 * np.random.random() - 1) for n in range(dim)])
    
    # Normalize the vector to make it unitary
    return vector / np.linalg.norm(vector)

def random_basis(n_qubits = 2, rank = None):
    # Generates a random basis with the specified rank.

    # Parameters:
    # n_qubits (int): Number of qubits, determines the dimension of the vectors (2**n_qubits).
    # rank (int, optional): The number of vectors in the basis. If None, a random rank is chosen.

    # Returns:
    # np.ndarray: Array of random unitary vectors of size 'rank'.

    dim = 2**n_qubits  # Dimension of the Hilbert space

    # If rank is not provided, choose a random value between 1 and the dimension of the space
    if rank is None:
        rank = random.randint(1, dim)

    # Generate 'rank' number of random unitary vectors
    return np.array([random_vector(n_qubits = n_qubits) for n in range(rank)])

def gram_schmidt(V):
    # Applies the Gram-Schmidt algorithm to orthogonalize a set of complex vectors.

    # Parameters:
    # V (np.ndarray): A matrix where each row represents a vector to orthogonalize.

    # Returns:
    # Q (np.ndarray): A matrix of orthonormal vectors where each row is an orthogonalized vector.

    n = V.shape[0]  # Number of vectors
    Q = np.zeros_like(V, dtype = complex)  # Initialize the orthogonal matrix

    for i in range(n):
        vi = V[i]  # Select the i-th vector
        
        # Subtract the projection of vi onto previous orthogonal vectors
        for j in range(i):
            vi -= np.dot(np.vdot(Q[j], vi), Q[j])
        
        # Normalize the orthogonal vector
        Q[i] = vi / np.linalg.norm(vi)

    return Q

def projector(vector):
    # Calculates the projection matrix for a given vector.

    # Parameters:
    # vector (np.ndarray): A non-zero vector.

    # Returns:
    # np.ndarray: The projection matrix corresponding to the input vector.

    vector = vector.reshape(-1, 1)  # Reshape the vector into a column
    
    # Compute the outer product of the vector with its conjugate transpose
    outer_product = np.dot(vector, np.conjugate(vector).T)
    return outer_product / np.dot(np.conjugate(vector).T, vector)

def random_state(n_qubits = 2, rank = None):
    # Generates a random quantum state (density matrix) of a given rank.
    
    # Parameters:
    # n_qubits (int): The number of qubits, determines the dimension of the state (2**n_qubits).
    # rango (int): The rank of the density matrix (number of vectors in the random basis).
    
    # Returns:
    # np.ndarray: A random density matrix of size (2**n_qubits, 2**n_qubits).

    # Generate an orthonormal basis using Gram-Schmidt process on a random basis
    basis = gram_schmidt(random_basis(n_qubits = n_qubits, rank = rank))
    rank  = basis.shape[0]  # Determine the rank based on the basis generated

    # Initialize the state as zero
    state = 0

    # Loop through the rank to build the random density matrix
    for n in range(rank):
        projection  = projector(basis[n])  # Create projector for each basis vector
        eigvalue    = random.random()  # Generate a random eigenvalue in the range [0, 1]
        state       = state + eigvalue * projection  # Add the scaled projector to the state

    # Make sure the matrix is Hermitian by symmetrizing it
    state = (state + np.conjugate(state).T) / 2

    # Normalize the state so that its trace equals 1
    return state / np.trace(state)

def werner_state(p = None):
    # Generates a Werner state, a mixture of a Bell state and the maximally mixed state.

    # Parameters:
    # p (float, optional): Probability weight for the Bell state. If None, a random value is used.

    # Returns:
    # tuple: Probability p and the Werner state matrix.

    bell_state = 1 / np.sqrt(2) * np.array([0, 1, -1, 0]).reshape(-1, 1)
    bell_state = projector(bell_state)

    # If no probability is provided, generate a random one
    if p == None:
        p = random.random()
        
    # Create the Werner state
    w_state = p * bell_state + (1 - p) / 4 * np.eye(4)
    return w_state, p

def state_evol(rho_initial, time, Hamiltonian):
    # Evolves a quantum state (density matrix) over time using unitary evolution governed by a Hamiltonian.
    
    # Parameters:
    # rho_initial (np.ndarray): The initial density matrix of the system.
    # time (float): The time over which the system evolves.
    # Hamiltonian (np.ndarray): The Hamiltonian governing the evolution of the system.
    
    # Returns:
    # np.ndarray: The final density matrix after the evolution, normalized to have trace 1.

    # Calculate the unitary evolution operator U = exp(-i * time * Hamiltonian)
    U = expm(-1j * time * Hamiltonian)
    
    # Evolve the state: rho_final = U * rho_initial * U^dagger
    rho_final = U @ rho_initial @ np.conjugate(U).T

    # Normalize the state so that the trace remains 1
    return rho_final / np.trace(rho_final)

def square_root(matrix=torch.Tensor):
    # Computes the square root of a matrix using eigendecomposition.

    # Parameters:
    # matrix (torch.Tensor): The input matrix.

    # Returns:
    # torch.Tensor: The square root of the input matrix.

    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    sqrt_eigenvalues = torch.diag(torch.sqrt(eigenvalues))  # Square root of eigenvalues
    
    # Reconstruct the matrix using the square root of eigenvalues
    return eigenvectors @ sqrt_eigenvalues @ torch.linalg.inv(eigenvectors)

def fidelity(rho, sigma, lib="torch"):
    # Computes the fidelity between two density matrices rho and sigma.

    # Parameters:
    # rho (np.ndarray or torch.Tensor): First density matrix.
    # sigma (np.ndarray or torch.Tensor): Second density matrix.
    # lib (str): Specifies the library to use ('numpy' or 'torch').

    # Returns:
    # float: Fidelity value between rho and sigma.

    if lib == "numpy":
        # Using numpy for the fidelity calculation
        sqrt_rho = sqrtm(rho)  # Matrix square root of rho
        fid_value = np.dot(sqrt_rho, np.dot(sigma, sqrt_rho))  # sqrt(rho) * sigma * sqrt(rho)
        fid_value = sqrtm(fid_value)  # Taking the square root of the resulting matrix
        
        return np.real(np.trace(fid_value))**2  # Fidelity is the squared trace of this matrix

    elif lib == "torch":
        # Using torch for the fidelity calculation
        sqrt_rho = square_root(rho)  # Matrix square root of rho
        fid_value = torch.matmul(sqrt_rho, torch.matmul(sigma, sqrt_rho))  # sqrt(rho) * sigma * sqrt(rho)
        fid_value = square_root(fid_value)  # Taking the square root of the resulting matrix
        
        return torch.real(torch.trace(fid_value))**2  # Fidelity is the squared trace of this matrix

    else:
        # Handle invalid library argument
        raise ValueError("Invalid value for 'lib'. Use 'torch' or 'numpy'.")
    
def partial_trace(rho, dim_A, dim_B, subsystem='B'):
    # Computes the partial trace of a density matrix over one of the subsystems.
    
    # Parameters:
    # rho (np.ndarray): The density matrix of the composite system.
    # dim_A (int): Dimension of subsystem A.
    # dim_B (int): Dimension of subsystem B.
    # subsystem (str): Specifies the subsystem to trace out ('A' or 'B').

    # Returns:
    # np.ndarray: The reduced density matrix after tracing out the specified subsystem.

    if subsystem == 'A':
        dims = (dim_A, dim_B)
    elif subsystem == 'B':
        dims = (dim_B, dim_A)
    else:
        raise ValueError("Subsystem must be 'A' or 'B'")

    # Reshape the density matrix for partial tracing
    rho_reshaped = rho.reshape(dims + dims)
    
    # Perform the partial trace over the chosen subsystem
    if subsystem == 'A':
        reduced_rho = np.trace(rho_reshaped, axis1=0, axis2=2)
    elif subsystem == 'B':
        reduced_rho = np.trace(rho_reshaped, axis1=1, axis2=3)
    
    return reduced_rho

def partial_transpose(rho, dims, subsystem):
    # Computes the partial transpose of a bipartite quantum state.
    
    # Parameters:
    # rho (np.ndarray): Density matrix of the bipartite system.
    # dims (tuple): Dimensions of the two subsystems (dim_A, dim_B).
    # subsystem (int): Specifies the subsystem for the partial transpose (0 for A, 1 for B).
    
    # Returns:
    # np.ndarray: The density matrix after applying the partial transpose.
    
    # Unpack dimensions of the subsystems
    dim_A, dim_B = dims
    rho_shape = rho.shape
    
    # Verify that the density matrix has the correct shape for the bipartite system
    assert rho_shape == (dim_A * dim_B, dim_A * dim_B)
    
    # Reshape the matrix to access subsystems separately
    rho_reshaped = rho.reshape([dim_A, dim_B, dim_A, dim_B])

    # Apply the partial transpose based on the specified subsystem
    if subsystem == 0:
        # Partial transpose on subsystem A
        rho_transposed = np.transpose(rho_reshaped, (2, 1, 0, 3))
    elif subsystem == 1:
        # Partial transpose on subsystem B
        rho_transposed = np.transpose(rho_reshaped, (0, 3, 2, 1))
    else:
        # Handle invalid subsystem input
        raise ValueError("Subsystem must be 0 (for A) or 1 (for B).")

    # Reshape the result back to the original matrix shape
    rho_partial_transpose = rho_transposed.reshape([dim_A * dim_B, dim_A * dim_B])
    
    return rho_partial_transpose