import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from band_structure_1d import MUBQBandStructure
import operator

sys_params = dict(
    X_gridDIM=512,

    # the lattice constant is 2 * X_amplitude
    X_amplitude=4.,

    # Lattice height
    V0=0.37,

    # the kinetic energy
    K=lambda self, p: 0.5 * p ** 2,

    # Mathieu-type periodic system
    V=lambda self, x: -self.V0 * (1 + np.cos(np.pi * (x + self.X_amplitude) / self.X_amplitude))
)

# initialize the system
qsys = MUBQBandStructure(**sys_params)

# how many eV is in 1 a.u. of energy
au2eV = 27.

# range of bloch vectors to compute the band structure
k_ampl = np.pi / qsys.X_amplitude
#K = np.linspace(-0.5, 0.5, 200)
#K = np.linspace(0., 0.5, 100)
K = np.linspace(-0.30, 0.30, 150)
dK = K[1] - K[0]

#for epsilon in qsys.get_band_structure(k_ampl * K, 4):
    #plt.plot(K, au2eV * 1. / np.gradient(np.gradient(epsilon)))
    #plt.plot(K, au2eV * 1. / np.gradient(np.gradient(epsilon, dK), dK), '-')
    #plt.plot(K, au2eV * epsilon)

"""
def J(x, qsys):
    K = np.linspace(x[0], x[1], 200)
    dK = K[1] - K[0]
    band_energy = (linalg.eigvalsh(qsys.get_hamiltonian(_)) for _ in K)
    band_energy = map(operator.itemgetter(0), band_energy)
    M = np.gradient(np.gradient(band_energy, dK), dK)
    #indx = (M < 0.)
    #negative = M[indx]
    #positive = M[np.logical_not(indx, out=indx)]

    #y = np.abs(negative + positive).sum()
    #print y
    y = abs(M.sum())
    print x, y
    return y

from scipy.optimize import minimize
result = minimize(J, [0.1, 0.3], args=(qsys,))
"""

band_energies = []
eigenstates = []

for k_val in K:
    vals, vecs = linalg.eigh(qsys.get_hamiltonian(k_val))

    # extract only first n eigenvectors
    #vecs = vecs[:, :n]
    #vals = vals[:n]

    band_energies.append(vals[1])
    eigenstates.append(vecs[:, 1])

effective_mass = 1./np.gradient(np.gradient(band_energies, dK), dK)


m, psi_init = max(zip(effective_mass, eigenstates))

print "effective mass ", m

"""
negeative_wave_function = 0.
positive_wave_function = 0.

for mass, psi in zip(effective_mass, eigenstates):
    if mass < 0:
        negeative_wave_function += psi
    else:
        positive_wave_function += psi

psi_init = -negeative_wave_function + positive_wave_function
"""

"""
plt.subplot(221)
plt.title('Effective mass')
plt.plot(K, effective_mass)

plt.subplot(222)
plt.title('Positive wavefunction')
plt.plot(qsys.X_range, positive_wave_function)

plt.subplot(223)
plt.title('Negative wavefunction')
plt.plot(qsys.X_range, negeative_wave_function)

plt.subplot(224)
plt.title("Sum")
plt.plot(qsys.X_range, positive_wave_function + negeative_wave_function)

plt.show()
"""

#plt.title("Reproduction of Fig. 1 from M. Wu et al. Phys. Rev A 91, 043839 (2015)")
#plt.xlabel("$k$ (units of $2\pi/ a_0$)")
#plt.ylabel('$\\varepsilon(k)$ (eV)')

if __name__=='__main__':
    # load tools for creating animation
    import sys
    import matplotlib

    if sys.platform == 'darwin':
        # only for MacOS
        matplotlib.use('TKAgg')

    import matplotlib.animation
    import matplotlib.pyplot as plt

    from AccurateWigner.rho_vneumann_cuda_1d import RhoVNeumannCUDA1D

    class VisualizeDynamicsPhaseSpace:
        """
        Class to visualize the Wigner function function dynamics in phase space.
        """

        def __init__(self, fig):
            """
            Initialize all propagators and frame
            :param fig: matplotlib figure object
            """
            #  Initialize systems
            self.set_quantum_sys()

            #################################################################
            #
            # Initialize plotting facility
            #
            #################################################################

            self.fig = fig

            ax = fig.add_subplot(121)

            ax.set_title('Wigner function, $W(x,p,t)$')
            extent = [
                self.quant_sys.X_wigner.min(), self.quant_sys.X_wigner.max(),
                self.quant_sys.P_wigner.min(), self.quant_sys.P_wigner.max()
            ]

            # import utility to visualize the wigner function
            from AccurateWigner.wigner_normalize import WignerNormalize

            # generate empty plot
            self.img = ax.imshow(
                [[]],
                extent=extent,
                origin='lower',
                interpolation='nearest',
                cmap='seismic',
                norm=WignerNormalize(vmin=-0.1, vmax=0.1)
            )

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

            #ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])

            ax = self.fig.add_subplot(122)

            self.lines_wigner_marginal, = ax.plot(
                [self.quant_sys.X_wigner.min(), self.quant_sys.X_wigner.max()], [0, 0.3],
                'r', label='Wigner marginal'
            )
            self.lines_rho, = ax.plot(
                [self.quant_sys.X.min(), self.quant_sys.X.max()], [0, 0.3],
                'b', label='Density matrix diagonal'
            )
            ax.legend()

            ax.set_xlabel('X (a.u.)')
            ax.set_ylabel('Probability')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            # Create propagator
            self.quant_sys = RhoVNeumannCUDA1D(
                t=0.,
                dt=0.01,

                X_gridDIM=512,
                X_amplitude=4.,

                # Lattice height
                V0 = 0.37,

                # kinetic energy part of the hamiltonian
                K="0.5 * P * P",

                # potential energy part of the hamiltonian
                V="-V0 * (1 + cos(M_PI * (X + X_amplitude) / X_amplitude))",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",

                diff_V="V0 * M_PI / X_amplitude * sin(M_PI * (X + X_amplitude) / X_amplitude)"

            )

            # set randomised initial condition
            self.quant_sys.set_rho(
                psi_init[:, np.newaxis] * psi_init[np.newaxis, :].conj()
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.img.set_array([[]])
            self.lines_wigner_marginal.set_data([], [])
            self.lines_rho.set_data([], [])
            return self.img, self.lines_wigner_marginal, self.lines_rho

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the wave function and then get the Wigner function
            W = self.quant_sys.propagate(100).get_wignerfunction().get()

            self.img.set_array(W.real)

            x_marginal = W.sum(axis=0).real
            x_marginal *= self.quant_sys.dP_wigner

            self.lines_wigner_marginal.set_data(self.quant_sys.X_wigner, x_marginal)
            self.lines_rho.set_data(
                self.quant_sys.X, self.quant_sys.rho.get().diagonal().real
            )

            return self.img, self.lines_wigner_marginal, self.lines_rho


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=np.arange(100), init_func=visualizer.empty_frame, repeat=True, blit=True
    )

    plt.show()

    # extract the reference to quantum system
    quant_sys = visualizer.quant_sys

    # Analyze how well the energy was preserved
    h = np.array(quant_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %f percent" % ((1. - h.min() / h.max()) * 100)
    )

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = dt * np.arange(len(quant_sys.X_average)) + dt

    plt.subplot(131)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='$\\langle -\\partial V/\\partial x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, h)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()