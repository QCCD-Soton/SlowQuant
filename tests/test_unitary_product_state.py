# type: ignore
import numba as nb
import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allstatetransfer as allstlr
import slowquant.unitary_coupled_cluster.linear_response.naive as naivelr
from slowquant.unitary_coupled_cluster.operator_state_algebra import build_operator_matrix
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


def test_ups_naivelr() -> None:
    """Test LiH UCCSD(2,2) LR."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    WF = WaveFunctionUPS(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("BFGS", True)
    LR = naivelr.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.129476) < 10**-4
    assert abs(LR.excitation_energies[1] - 0.178749) < 10**-4
    assert abs(LR.excitation_energies[2] - 0.178749) < 10**-4
    assert abs(LR.excitation_energies[3] - 0.604681) < 10**-4
    assert abs(LR.excitation_energies[4] - 0.646707) < 10**-4
    assert abs(LR.excitation_energies[5] - 0.740632) < 10**-4
    assert abs(LR.excitation_energies[6] - 0.740632) < 10**-4
    assert abs(LR.excitation_energies[7] - 1.002914) < 10**-4
    assert abs(LR.excitation_energies[8] - 2.074822) < 10**-4
    assert abs(LR.excitation_energies[9] - 2.137193) < 10**-4
    assert abs(LR.excitation_energies[10] - 2.137193) < 10**-4
    assert abs(LR.excitation_energies[11] - 2.455191) < 10**-4
    assert abs(LR.excitation_energies[12] - 2.954372) < 10**-4
    osc_strengths = LR.get_oscillator_strength()
    assert abs(osc_strengths[0] - 0.049920) < 10**-3
    assert abs(osc_strengths[1] - 0.241184) < 10**-3
    assert abs(osc_strengths[2] - 0.241184) < 10**-3
    assert abs(osc_strengths[3] - 0.158045) < 10**-3
    assert abs(osc_strengths[4] - 0.166539) < 10**-3
    assert abs(osc_strengths[5] - 0.010379) < 10**-3
    assert abs(osc_strengths[6] - 0.010379) < 10**-3
    assert abs(osc_strengths[7] - 0.006256) < 10**-3
    assert abs(osc_strengths[8] - 0.062386) < 10**-3
    assert abs(osc_strengths[9] - 0.128862) < 10**-3
    assert abs(osc_strengths[10] - 0.128862) < 10**-3
    assert abs(osc_strengths[11] - 0.046007) < 10**-3
    assert abs(osc_strengths[12] - 0.003904) < 10**-3


def test_LiH_sto3g_allST():
    """Test LiH STO-3G all-statetransfer LR oscialltor strength."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.67 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)
    WF2 = WaveFunctionUPS(
        (2, 2),
        WF.c_mo,
        SQobj,
        "tUPS",
        ansatz_options={"n_layers": 1},
    )
    WF2.run_wf_optimization_1step("BFGS", False)
    # Linear Response
    LR = allstlr.LinearResponse(
        WF2,
        excitations="SD",
    )
    LR.calc_excitation_energies()

    thresh = 10**-3

    # Check excitation energies
    solutions = np.array(
        [
            0.1851181,
            0.24715136,
            0.24715136,
            0.6230648,
            0.85960395,
            2.07752209,
            2.13720198,
            2.13720198,
            2.55113802,
        ]
    )

    assert np.allclose(LR.excitation_energies, solutions, atol=thresh)

    # Get oscillator strength for each excited state
    osc_strengths = LR.get_oscillator_strength()
    assert abs(osc_strengths[0] - 0.06668878) < thresh
    assert abs(osc_strengths[1] - 0.33360367) < thresh
    assert abs(osc_strengths[2] - 0.33360367) < thresh
    assert abs(osc_strengths[3] - 0.30588158) < thresh
    assert abs(osc_strengths[4] - 0.02569977) < thresh
    assert abs(osc_strengths[5] - 0.06690658) < thresh
    assert abs(osc_strengths[6] - 0.13411942) < thresh
    assert abs(osc_strengths[7] - 0.13411942) < thresh
    assert abs(osc_strengths[8] - 0.04689274) < thresh


def test_ups_water_44() -> None:
    """Test a larger active space."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """O   0.0  0.0           0.1035174918;
    H   0.0  0.7955612117 -0.4640237459;
    H   0.0 -0.7955612117 -0.4640237459;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    WF = WaveFunctionUPS(
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "fUCCSD",
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("bfgs", True)
    # Changed threshold from 10-8 to 10-6, due to loss of precision in optimizer.
    # Change back again when optimization code has been made more stable.
    assert abs(WF.energy_elec - -84.00619955119978) < 10**-6


def test_saups_h2_3states() -> None:
    """This should give exactly the same as FCI.

    Since all states, are includes in the subspace expansion.
    """
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H   0.0  0.0  0.0;
           H   0.0  0.0  0.735;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    WF = WaveFunctionSAUPS(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
                [1],
            ],
            [
                ["1100"],
                ["1001", "0110"],
                ["0011"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step("BFGS", True)

    assert abs(WF.excitation_energies[0] - 0.974553) < 10**-6
    assert abs(WF.excitation_energies[1] - 1.632364) < 10**-6
    osc = WF.get_oscillator_strenghts()
    assert abs(osc[0] - 0.8706) < 10**-3
    assert abs(osc[1] - 0.0) < 10**-3


def test_saups_h3_3states() -> None:
    """Test a system where the subspace is not everything."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H   -0.45  -0.3897114317  0.0;
           H   0.45  -0.3897114317  0.0;
           H   0.0  0.3897114317  0.0;""",
        distance_unit="angstrom",
        molecular_charge=1,
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    WF = WaveFunctionSAUPS(
        (2, 3),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["110000"],
                ["100100", "011000"],
                ["100001", "010010"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 2, "skip_last_singles": True},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_2step("BFGS", True, tol=10**-8)

    assert abs(WF.excitation_energies[0] - 0.838466) < 10**-6
    assert abs(WF.excitation_energies[1] - 0.838466) < 10**-6
    osc = WF.get_oscillator_strenghts()
    assert abs(osc[0] - 0.7569) < 10**-3
    assert abs(osc[1] - 0.7569) < 10**-3


def test_sa_doubles() -> None:
    """Test spin-adapted double excitations."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0 0 0;
        H 1.6 0 0""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    WF = WaveFunctionUPS(
        (4, 6),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        ansatz="fUCC",
        ansatz_options={"n_layers": 1, "SAS": True, "SAD": True},
    )
    WF.run_wf_optimization_1step("BFGS")
    assert abs(WF.energy_elec - -8.874521029611891) < 10**-8


def test_SA_sa_doubles() -> None:
    """Test spin-adapted double excitations."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0 0 0;
        H 1.6 0 0""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    WF = WaveFunctionSAUPS(
        (4, 6),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        ([[1]], [["111100000000"]]),
        ansatz="SAfUCCSD",
        ansatz_options={"n_layers": 1, "SAS": True, "SAD": True},
    )
    WF.run_wf_optimization_1step("BFGS")
    assert abs(WF.energy_states[0] - -8.874521029611891) < 10**-8


def test_ups_water_44_threaded() -> None:
    """Test a larger active space."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """O   0.0  0.0           0.1035174918;
    H   0.0  0.7955612117 -0.4640237459;
    H   0.0 -0.7955612117 -0.4640237459;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    nb.set_num_threads(2)
    WF = WaveFunctionUPS(
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "fUCCSD",
        ansatz_options={},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("SLSQP", True, tol=10**-8)
    assert abs(WF.energy_elec - -83.97256228053688) < 10**-8
    nb.set_num_threads(1)


def test_saups_h3_3states_threaded() -> None:
    """Test a system where the subspace is not everything."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H   -0.45  -0.3897114317  0.0;
           H   0.45  -0.3897114317  0.0;
           H   0.0  0.3897114317  0.0;""",
        distance_unit="angstrom",
        molecular_charge=1,
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    nb.set_num_threads(2)
    WF = WaveFunctionSAUPS(
        (2, 3),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["110000"],
                ["100100", "011000"],
                ["100001", "010010"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 2, "skip_last_singles": True},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_2step("BFGS", True, tol=10**-8)

    assert abs(WF.excitation_energies[0] - 0.838466) < 10**-6
    assert abs(WF.excitation_energies[1] - 0.838466) < 10**-6
    osc = WF.get_oscillator_strenghts()
    assert abs(osc[0] - 0.7569) < 10**-3
    assert abs(osc[1] - 0.7569) < 10**-3
    nb.set_num_threads(1)


def test_pptUPS_h2o() -> None:
    """Test pptUPS on H2O(4,4)."""
    # Define molecule: H2O
    atom = """O   0.0  0.0           0.1035174918;
        H   0.0  0.7955612117 -0.4640237459;
        H   0.0 -0.7955612117 -0.4640237459;"""
    basis = "sto-3g"

    # HF
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        atom,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set(basis)
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    mo_coeff = SQobj.hartree_fock.mo_coeff
    integral_generator = SQobj

    # pp-tUPS
    WF = WaveFunctionUPS(
        (4, 4),  # active space
        mo_coeff,
        integral_generator,
        "tUPS",  # our circuit Ansatz
        ansatz_options={"n_layers": 1, "do_pp": True},  # we use 1 layer
    )

    WF.run_wf_optimization_1step("bfgs", orbital_optimization=False)

    assert abs(WF.energy_elec + 83.96387402720552) < 10**-6


def test_total_spin() -> None:
    """Test pp in ups and circuit and its translation from ups to circuit."""
    # Define molecule: H2O
    atom = """O   0.0  0.0           0.1035174918;
        H   0.0  0.7955612117 -0.4640237459;
        H   0.0 -0.7955612117 -0.4640237459;"""
    basis = "sto-3g"

    # HF
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        atom,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set(basis)
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    mo_coeff = SQobj.hartree_fock.mo_coeff
    integral_generator = SQobj

    WF = WaveFunctionUPS(
        (10, 7),  # active space
        mo_coeff,
        integral_generator,
        "tUPS",  # our circuit Ansatz
        ansatz_options={"n_layers": 1},  # we use 1 layer
        include_active_kappa=True,
    )

    assert WF.get_total_spin() == 0.0


def test_ups_qfim() -> None:
    """Test the UPS quantum Fisher information matrix against finite differences, LiH STO-3G."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    for cas, ansatz, options in (
        ((2, 2), "tUPS", {"n_layers": 1}),
        ((4, 3), "SDSfUCCSD", {"n_layers": 1}),
    ):
        WF = WaveFunctionUPS(
            cas,
            SQobj.hartree_fock.mo_coeff,
            SQobj,
            ansatz,
            ansatz_options=options,
        )
        n_params = WF.ups_layout.n_params
        n_det = len(WF.ci_coeffs)
        thetas = np.array([0.3 * np.sin(i + 1) - 0.1 * np.cos(2 * i + 1) for i in range(n_params)])
        WF.thetas = thetas.tolist()
        psi = np.copy(WF.ci_coeffs)
        # Derivatives of the state against central finite differences
        D = WF._calc_state_derivatives()
        step = 10**-5
        D_fd = np.zeros_like(D)
        for i in range(n_params):
            for sign in (1.0, -1.0):
                shifted = thetas.copy()
                shifted[i] += sign * step
                WF.thetas = shifted.tolist()
                D_fd[:, i] += sign * WF.ci_coeffs / (2 * step)
        WF.thetas = thetas.tolist()
        assert np.max(np.abs(D - D_fd)) < 10**-8
        # Normalisation gives <psi|d_i psi> = 0
        assert np.max(np.abs(np.matmul(D.T, psi))) < 10**-12
        # Energy gradient from the state derivatives, dE/dtheta_i = 2 <d_i psi|H|psi>
        H = build_operator_matrix(WF._get_hamiltonian(), WF.ci_info, do_unsafe=True)  # type: ignore
        gradient = WF._calc_gradient_optimization(thetas.tolist(), True, False)
        assert np.max(np.abs(2 * np.matmul(D.T, np.matmul(H, psi)) - gradient)) < 10**-10
        # QFIM is symmetric positive semi-definite with rank at most min(n_params, n_det - 1)
        F = WF.get_qfim()
        assert np.max(np.abs(F - F.T)) < 10**-14
        assert np.max(np.abs(F - 4 * WF.get_qfim(as_metric=True))) < 10**-14
        assert np.max(np.abs(F - WF._calc_qfim_optimization(thetas.tolist(), True, False))) < 10**-14
        eigvals = np.linalg.eigvalsh(F)
        assert eigvals[0] > -(10**-10)
        rank = int(np.sum(eigvals > 10**-10 * eigvals[-1]))
        assert rank <= min(n_params, n_det - 1)
        if ansatz == "SDSfUCCSD":
            # The full CAS(4,3) space is reached, so the bound is saturated
            assert rank == n_det - 1
        # Fidelity, |<psi(theta)|psi(theta + d)>|^2 = 1 - d^T F d / 4 + O(|d|^3)
        d = 10**-3 * np.array([np.cos(3 * i + 2) for i in range(n_params)])
        WF.thetas = (thetas + d).tolist()
        infidelity = 1 - np.dot(psi, WF.ci_coeffs) ** 2
        WF.thetas = thetas.tolist()
        assert abs(infidelity - 0.25 * np.dot(d, np.matmul(F, d))) < 10**-8
