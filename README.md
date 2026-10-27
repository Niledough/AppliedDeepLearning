# Applied Deep Learning Project WS 2026

## Project Outline

For this "Bring your own method" type project, i want to investigate Port-Hamiltonian Neural Networks (pHNN) as introduced in [ref 1][1] (HNN) and extended to be able to capture damping and forcing as well in [ref 2][2] (pHNN). 

### Motivation for Port-Hamilton Neural Networks

Many advanced control algorithms, like Model Predictive Control, rely heavily on an accurate state model of the system that is to be controlled. Historically, the modeling required careful parameter determination, which often requires substantial efforts, like experiments in wind tunnels. While generic black-box neural networks (NN) are a promising approach to predicting system dynamics directly from the data, they don't provide any physical guarantees. As a consequence of this, huge amounts of data are required to get to an acceptable level of adherence to the physical laws, which still doesn't necessarily generalize well to unseen regions in state-space. This is a problem that physics-informed neuronal networks (PINN) try to solve through various mechanisms ([ref 4][4]). Often, the sought-after guarantees, come in the form of conservation laws like energy conservation for a closed system. Generally, this conserved quantity is characterized by the Hamiltonian, which is the total energy in the system as a function of the state. One approach making use of this concept is Hamiltonian neuronal networks (HNN), which aim to predict the Hamiltonian for a given state. While more generic PINN approaches would, e.g evaluate the loss with regard to how accurately the Hamiltonian is adhered to, HNNs conserve energy exactly by construction. As opposed to this, conventional black-box NN approaches would just fit the NN to the datapoints without any consideration of energy preservation. The HNN framework has been extended to port-hamiltonian NNs (pHNN) which make it possible to also capture energy dissipation and unknown time-dependent control forces. More current research is concerned with incorporating known forcing terms into the formulation ([ref 3][3]), which is a crucial stepping stone for utilizing the inverse model as a controller.

### pHNN Formulation

First, the generic HNN formulation following [ref 1][1] for a system with states $\boldsymbol{p}$ and $\boldsymbol{q}$. Without proof, assume that the Euler-Lagrange Equation
$$
\begin{equation}
{\displaystyle {\frac {\partial {\mathcal {L}}}{\partial {\boldsymbol {q}}}}-{\frac {d}{dt}}{\frac {\partial {\mathcal {L}}}{\partial {\dot {\boldsymbol {q}}}}}=0}
\end{equation}
$$
yields Hamilton's equation:
$$
\begin{equation}
{\displaystyle {\frac {\mathrm {d} {\boldsymbol {q}}}{\mathrm {d} t}}={\frac {\partial {\mathcal {H}}}{\partial {\boldsymbol {p}}}},\quad {\frac {\mathrm {d} {\boldsymbol {p}}}{\mathrm {d} t}}=-{\frac {\partial {\mathcal {H}}}{\partial {\boldsymbol {q}}}}.}
\end{equation}
$$
From (2) we can see that moving in the direction $\bold{S}_{\mathcal{H}} = \left(\frac{\partial \mathcal{H}}{\partial \boldsymbol{p}}, -\frac{\partial \mathcal{H}}{\partial \boldsymbol{q}}\right)$, the so-called symplectic gradient, gives the time evolution of the system. In contrast to the ordinary gradient, which changes $\mathcal{H}$ as quickly as possible, moving along the symplectic gradient keeps $\mathcal{H}$ exactly constant:
$$
\begin{equation}
\frac{d\mathcal{H}}{dt} = \frac{\partial \mathcal{H}}{\partial \boldsymbol{q}}^{\top}\dot{\boldsymbol{q}} + \frac{\partial \mathcal{H}}{\partial \boldsymbol{p}}^{\top}\dot{\boldsymbol{p}} = \frac{\partial \mathcal{H}}{\partial \boldsymbol{q}}^{\top}\frac{\partial \mathcal{H}}{\partial \boldsymbol{p}} - \frac{\partial \mathcal{H}}{\partial \boldsymbol{p}}^{\top}\frac{\partial \mathcal{H}}{\partial \boldsymbol{q}} = 0.
\end{equation}
$$

In order to ensure that $(2)$ i.e. energy conservation is respected, the model is trained to output a single scalar $\mathcal{H}_\theta(\boldsymbol{q}, \boldsymbol{p})$ which is then differentiated with respect to its input states ($\boldsymbol{q}, \boldsymbol{p}$). This gradient can then be compared to the temporal derivative of the states to yield the cost function $(4)$.
$$
\begin{equation}
\mathcal{L}_{\text{HNN}} = \left\| \frac{\partial \mathcal{H}_\theta}{\partial \boldsymbol{p}} - \dot{\boldsymbol{q}} \right\|_2^2 + \left\| \frac{\partial \mathcal{H}_\theta}{\partial \boldsymbol{q}} + \dot{\boldsymbol{p}} \right\|_2^2.
\end{equation}
$$
Since (3) holds for any scalar function, the learned $\mathcal{H}_\theta$ is conserved by construction, independent of the training outcome. Training only determines how closely $\mathcal{H}_\theta$ resembles the true energy of the system.

The HNN assumes a closed system and therefore cannot represent energy dissipation or external forces. [ref 2][2] address this by embedding the port-Hamiltonian formulation instead:
$$
\begin{equation}
\begin{bmatrix} \dot{\boldsymbol{q}} \\ \dot{\boldsymbol{p}} \end{bmatrix} =
\left( \begin{bmatrix} \boldsymbol{0} & \boldsymbol{I} \\ -\boldsymbol{I} & \boldsymbol{0} \end{bmatrix} - \begin{bmatrix} \boldsymbol{0} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{N} \end{bmatrix} \right)
\begin{bmatrix} \partial \mathcal{H}/\partial \boldsymbol{q} \\ \partial \mathcal{H}/\partial \boldsymbol{p} \end{bmatrix}
+ \begin{bmatrix} \boldsymbol{0} \\ \boldsymbol{F}(t) \end{bmatrix},
\end{equation}
$$
where $\boldsymbol{N} \succeq 0$ models damping and $\boldsymbol{F}(t)$ an external time-dependent force. Three components are learned: a network for the Hamiltonian $\mathcal{H}_{\theta_1}(\boldsymbol{q}, \boldsymbol{p})$, a network for the force $\boldsymbol{F}_{\theta_2}(t)$, and a single trainable parameter for the damping $\boldsymbol{N}_{\theta_3}$. Energy is no longer conserved, but it can only change through these two terms:
$$
\begin{equation}
\frac{d\mathcal{H}}{dt} = -\frac{\partial \mathcal{H}}{\partial \boldsymbol{p}}^{\top} \boldsymbol{N} \frac{\partial \mathcal{H}}{\partial \boldsymbol{p}} + \frac{\partial \mathcal{H}}{\partial \boldsymbol{p}}^{\top} \boldsymbol{F}(t),
\end{equation}
$$
i.e. damping can only remove energy, while the force injects or extracts power. The loss extends (4) by $L_1$ penalties on force and damping, which discourage the network from attributing dynamics to spurious forcing or damping terms:
$$
\begin{equation}
\mathcal{L}_{\text{pHNN}} = \left\| \hat{\dot{\boldsymbol{q}}} - \dot{\boldsymbol{q}} \right\|_2^2 + \left\| \hat{\dot{\boldsymbol{p}}} - \dot{\boldsymbol{p}} \right\|_2^2 + \lambda_F \left\| \boldsymbol{F}_{\theta_2} \right\|_1 + \lambda_N \left\| \boldsymbol{N}_{\theta_3} \right\|_1.
\end{equation}
$$

## Scope of this project

This project aims to do system identification using a pHNN following [ref 2][2] and then try to improve on that baseline by experimenting with different network architectures. If the progress of the project admits, it would also be interesting to try to incorporate known forcing terms into that formulation to try to use it as a simple controller. Furthermore, degradation of the performance for different levels of "measurement" noise could be investigated, though no reference numbers for that exist in [ref 2][2].

### Dataset

As the phenomena that the pHNN wants to recover can often be described by simple equations, generating the data from simulations is a commonly used and very practical approach. In systems where energy is preserved, it is likely that oscillations occur, so driven, dampened oscillators lend themselves very well to demonstrating the capabilities of the method. Canonical examples of this are forced and dampened mass-spring systems and Duffing oscillators, which are both used for benchmarking in [ref 2][2]. Because both oscillators can be represented as Ordinary Differential Equations (ODEs), exact solutions can be determined through standard methods. Finally, sampling and possibly perturbing the signal with noise yields input ($x$) and output ($\dot{x}$) data for the Machine Learning (ML) problem.

## References

-   [1]: https://arxiv.org/abs/2107.08024	"Hamiltonian Neural Networks"

-   [2]: http://arxiv.org/abs/2107.08024	"Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems"

-   [3]: https://ar5iv.labs.arxiv.org/html/2401.09520	"Port-Hamiltonian Neural ODE Networks on Lie Groups For Robot Dynamics Learning and Control"

-   [4]: https://www.nature.com/articles/s42254-021-00314-5	"Physics-informed machine learning"