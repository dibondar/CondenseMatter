from AccurateWigner.rho_vneumann_cuda_1d import RhoVNeumannCUDA1D, SourceModule, np


class WignerCondenseMatter(RhoVNeumannCUDA1D):

    def __init__(self, **kwargs):

        # initialize the parent class
        RhoVNeumannCUDA1D.__init__(self, **kwargs)

        # check that the Bloch vector was specialized
        try:
            self.bloch_k
        except AttributeError:
            raise AttributeError("Bloch vector (bloch_k) was not specified")

        self.bloch_k = np.float64(self.bloch_k)

        # Compile the function used to apply bloch vector factors before and after propagation
        self.bloch_phase = SourceModule(
            self.bloch_phase_cuda_source.format(cuda_consts=self.cuda_consts)
        ).get_function("bloch_phase")

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final density matrix function is not normalized.
        :return: self.rho
        """
        # rho *= exp(-ikx) * exp(+ikx')
        self.bloch_phase(self.rho, -self.bloch_k, **self.rho_mapper_params  )

        RhoVNeumannCUDA1D.single_step_propagation(self)

        # rho *= exp(ikx) * exp(-ikx')
        self.bloch_phase(self.rho, self.bloch_k, **self.rho_mapper_params)

        return self.rho

    bloch_phase_cuda_source = """
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void bloch_phase (cuda_complex *rho, const double k)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double X_prime = dX * (i - 0.5 * X_gridDIM);

        rho[indexTotal] *= cuda_complex(cos(k * X), sin(k * X))
                            * cuda_complex(cos(k * X_prime), -sin(k * X_prime));
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(WignerCondenseMatter.__doc__)

    # load tools for creating animation
    import sys
    import matplotlib

    if sys.platform == 'darwin':
        # only for MacOS
        matplotlib.use('TKAgg')

    import matplotlib.animation
    import matplotlib.pyplot as plt


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
            """
            # Create propagator
            self.quant_sys = WignerCondenseMatter(
                t=0.,
                dt=0.01,

                X_gridDIM=1024,
                X_amplitude=10.,

                # randomized parameter
                omega_square=np.random.uniform(0.01, 0.1),

                # randomized parameters for initial condition
                sigma=np.random.uniform(0.5, 4.),
                p0=np.random.uniform(-1., 1.),
                x0=np.random.uniform(-1., 1.),

                # kinetic energy part of the hamiltonian
                K="0.5 * P * P",

                # potential energy part of the hamiltonian
                V="0.5 * omega_square * X * X",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",
                diff_V="omega_square * X",

                # Bloch vector
                bloch_k=0.,
            )

            # set randomised initial condition
            self.quant_sys.set_rho(
                "exp(-sigma * pow(X - x0, 2) + cuda_complex(0., p0) * X) *"
                "exp(-sigma * pow(X_prime - x0, 2) - cuda_complex(0., p0) * X_prime)"
            )
            """

            np.random.seed(5)

            sys_params = dict(
                t=0.,
                dt=0.01,

                X_gridDIM=1024,
                X_amplitude=10.,

                # randomized parameter
                V0=np.random.uniform(0.01, 0.1),

                # kinetic energy part of the hamiltonian
                K="0.5 * (P + bloch_k) * (P + bloch_k)",

                # potential energy part of the hamiltonian
                V="-V0 * (1 + cos(M_PI * (X + X_amplitude) / X_amplitude))",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",
                diff_V=" X",

                # Bloch vector
                bloch_k=0.01, #np.random.uniform(0., 1.),
            )

            # Create propagator
            self.quant_sys = WignerCondenseMatter(**sys_params)

            # Replace V and K by pythonic functions
            sys_params.update(
                K=lambda self, P: 0.5 * P * P,
                V=lambda self, x: -self.V0 * (1 + np.cos(np.pi * (x + self.X_amplitude) / self.X_amplitude))
            )

            # Find the Bloch state with bloch vector equals to bloch_k
            bloch_k = self.quant_sys.bloch_k
            from band_structure_1d import MUBQBandStructure, linalg

            _, phi = linalg.eigh(
                MUBQBandStructure(**sys_params).get_hamiltonian(bloch_k)
            )

            psi = phi[:, 2] * np.exp(1j * bloch_k * self.quant_sys.X)

            self.quant_sys.set_rho(
                psi[np.newaxis, :] * psi[:, np.newaxis].conj()
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
            W = self.quant_sys.propagate(50).get_wignerfunction().get()

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
