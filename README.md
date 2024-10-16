# DeepPhysics-Nets

![image](https://github.com/user-attachments/assets/cc247a81-6465-4cac-987f-45e9f454f77d)

Welcome to the **Quantum Information and Machine Learning** repository! This repository showcases various neural network models and physics-informed neural networks (PINNs) developed to solve problems at the intersection of machine learning and quantum information theory. These projects aim to demonstrate how machine learning techniques can be applied to advance quantum physics and address key challenges in quantum information processing.

## Projects Overview

1. **CNN-based Quantum Evolution**  
   This project presents a convolutional neural network (CNN)-based model that predicts the time evolution of a quantum state under the influence of any given Hamiltonian. By using convolutional layers and introducing a scalar parameter in the latent space, this model efficiently determines the quantum state's evolution after an arbitrary time. This model helps simulate and analyze dynamic quantum systems.

2. **Entanglement Classifier**  
   A neural network designed to classify whether a two-qubit quantum state is entangled or not. Entanglement certification is not an easy problem to tackle, so the idea behind this is to generate random states and label them with an entanglement criterion (PPT for this case). After that, the training stage updates parameters of the network that relate the quantum state to a scalar that tells us whether the state is entangled or not.

3. **Physics-Informed Neural Networks (PINNs) for Differential Equations in Physics**  
   This project leverages PINNs to solve some important differential equations in physics:
   - **Helmholtz Equation**: Simulates the spatial evolution of electromagnetic waves.
   - **Wave Equation**: Models the behavior of waves in different media.
   - **Second Harmonic Generation Equation**: Addresses the non-linear differential equation governing the second harmonic generation in optics.

## Repository Structure

Each project is contained within its own directory, with explanations and examples to guide you through their functionality and implementation. These examples highlight the power of machine learning to solve complex quantum and physical systems.

## Requirements

To run these projects, you'll need the following dependencies:

- **Python 3.8+**
- **Libraries**:
  - numpy
  - torch
  - matplotlib
  - random
  - tqdm
  - scipy
 
