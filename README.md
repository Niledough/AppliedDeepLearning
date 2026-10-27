# Applied Deep Learning Project WS 2026

## Project Outline

For this "Bring your own method" type project, i want to investigate Port-Hamiltonian Neural Networks as introduced [here](https://arxiv.org/abs/2107.08024) and applied to robotics [here](https://ar5iv.labs.arxiv.org/html/2401.09520). 

### Motivation for Port-Hamilton Neural Networks

Many advanced control algorithms like Model Predictive Control, rely heavily on a accurate state models of the system that is to be controlled. Historically, the modeling required careful parameter determination, which often requires substantial efforts like experiments in wind tunnels. While black-box neural networks (NN) are a promising approach to predicting system dynamics directly from the data, they don't provide any physical guarantees. As a consequence of this, huge amounts of data are required to get to an acceptable level of adherence to the physical laws, which still doesn't necessarily generalize well to unseen regions in state-space. This is a problem that physics informed neuronal networks (PINN) try to solve through various mechanisms. Often, the sought after guarantees, come in the form of conservation laws like energy conservation for a closed system. Generally this conserved quantity is characterized by the Hamiltonian, which is the total energy in the system as a function of the state. One approach making use of this concept are Hamiltonian neuronal networks (HNN), which aim to predict the hamiltonian for a given state. While more generic PINN approaches would e.g evaluate the loss with regards to how accurately the Hamiltonian is adhered to, HNNs conserve energy exactly by construction. As opposed to this, conventional black-box NN approaches would just fit the NN to the datapoints without any consideration of energy preservation. The HNN framework has been extended to so called [port-hamiltonian NNs](https://arxiv.org/abs/2107.08024) (pHNN) which make it possible to also capture energy dissipation and unknown time dependent control forces. Finally the pHNN formulation has been tweaked in [this paper]( https://arxiv.org/abs/2107.08024), to allow for system identification with known control forces. 

### Scope of this project

This project aims to implement a pHNN for known control forces and then try to improve on that baseline


$$
\frac{test}{this}
$$


