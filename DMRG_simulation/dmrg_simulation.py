from planted_solutions.CAS.dmrghandler.src.dmrghandler.qchem_dmrg_calc import single_qchem_dmrg_calc
from planted_solutions.CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import check_spin_symmetry, spinorbitals_to_orbitals
from planted_solutions.DMRG_simulation.validator.symmetry_check import check_permutation_symmetries_complex_orbitals
from planted_solutions.DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation
from planted_solutions.DMRG_simulation.format_dmrg.format_dmrg_param import get_dmrg_param, get_dmrg_process_param
import openfermion as of
import numpy as np
import planted_solutions.CAS.saveload_utils as sl
import planted_solutions.CAS.ferm_utils as feru
import planted_solutions.CAS.var_utils as varu
import sys
sys.path.append("../CAS/")


def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace


default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8


def get_ground_state_fci(mol, path):
    """
    Use the openfermion library to calculate the ground state FCI

    Args:
        mol: molecular file path

    Returns: ground state FCI and ground state energy

    """
    # The type of Hf is Fermion Operator
    Hf = sl.load_fermionic_hamiltonian(mol, prefix=path)
    spin_orb = of.count_qubits(Hf)
    print(type(Hf))

    # Get the chemist tbt
    Htbt = feru.get_chemist_tbt(Hf, spin_orb, spin_orb=True)

    # Get the one_body_correction
    one_body = varu.get_one_body_correction_from_tbt(Hf,
                                                     feru.get_chemist_tbt(Hf))
    onebody_matrix = feru.get_obt(one_body, n=spin_orb, spin_orb=True)
    onebody_tbt = feru.onebody_to_twobody(onebody_matrix)

    # Everything in two body tensor
    Htbt = np.add(Htbt, onebody_tbt)
    recombined = feru.get_ferm_op(Htbt, True)
    print(type(recombined))
    sparse = of.linalg.get_sparse_operator(recombined)
    ground_energy, gs = of.linalg.get_ground_state(
        sparse, initial_guess=None
    )
    return ground_energy, gs


def get_dmrg_energy(mol, dmrg_param):
    """
    Use DMRG to calculate the ground state and ground state energy
    Args:
        mol: molecular file path
        dmrg_param: DMRG parameters

    Returns:

    """
    # Load the hamiltonian
    Hf = sl.load_fermionic_hamiltonian(mol, prefix="../CAS/")
    spin_orb = of.count_qubits(Hf)


    # Htbt is in a^ a a^ a format
    Htbt = feru.get_chemist_tbt(Hf, spin_orb, spin_orb=True)

    # Get one body difference
    one_body = varu.get_one_body_correction_from_tbt(Hf,
                                                     feru.get_chemist_tbt(Hf))

    # Get the one body tensor from the difference
    onebody_matrix = feru.get_obt(one_body, n=spin_orb, spin_orb=True)

    # Check symmetries
    print("Check the Spin Symmetry", check_spin_symmetry(onebody_matrix, Htbt))
    print("Check permutation Symmetry",
          check_permutation_symmetries_complex_orbitals(onebody_matrix, Htbt))

    # Get the correct permutation of tensors for Block2
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(onebody_matrix, Htbt, spin_orb))
    print("Check permutation Symmetry After correction",
          check_permutation_symmetries_complex_orbitals(
              one_body_tensor, two_body_tensor))

    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

    print("Spin symmetry broken", spin_symm_broken)
    result = single_qchem_dmrg_calc(
        new_one_body_tensor, new_two_body_tensor, dmrg_param)
    return result


if __name__ == "__main__":
    # sys.settrace(trace)
    mol = 'h4' if len(sys.argv) < 2 else sys.argv[1]
    ground_energy, ground_state = get_ground_state_fci(mol, path="../CAS/")
    Hf = sl.load_fermionic_hamiltonian(mol, prefix="../CAS/")
    spin_orb = of.count_qubits(Hf)
    print(spin_orb)
    num_orbitals = spin_orb // 2
    num_electrons = 4
    num_spin_orbitals = spin_orb
    basis = "sto3g"
    num_unpaired_electrons = 0
    charge = 0
    multiplicity = 1 + (num_electrons % 2) * 2

    init_state_bond_dimension = 50
    max_num_sweeps = 200
    energy_convergence_threshold = 1e-8
    sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
    sweep_schedule_noise = default_sweep_schedule_noise
    sweep_schedule_davidson_threshold = (
        default_sweep_schedule_davidson_threshold
    )
    nuc_rep_energy = 0
    dmrg_process_param = get_dmrg_process_param(
        init_state_bond_dimension, max_num_sweeps, energy_convergence_threshold,
        sweep_schedule_bond_dims, sweep_schedule_noise,
        sweep_schedule_davidson_threshold)
    dmrg_param = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons,
        multiplicity, dmrg_process_param)
    dmrg_result = get_dmrg_energy(mol, dmrg_param)
    print("FCI groundstate energy:", ground_energy)
    print("DMRG groundstate energy:", dmrg_result["dmrg_ground_state_energy"])
    print("Energy difference:", dmrg_result["dmrg_ground_state_energy"] - ground_energy)





# def get_dmrg_with_phy(mol, dmrg_param):
#     """
#
#     Args:
#         mol:
#         dmrg_param:
#
#     Returns:
#
#     """
#     # Load the hamiltonian
#     Hf = sl.load_fermionic_hamiltonian(mol, prefix="./")
#     spin_orb = of.count_qubits(Hf)
#
#     phy_tbt = feru.get_physics_tbt(Hf, spin_orb, spin_orb=True)
#
#     # One body physics
#     one_body_phy = varu.get_one_body_correction_from_tbt(Hf,
#                                                          feru.get_physics_tbt(
#                                                              Hf))
#
#     # Get the one body tensor from the difference
#     onebody_matrix = feru.get_obt(one_body_phy, n=spin_orb, spin_orb=True)
#
#     # Check symmetries
#     print("Check permutation Symmetry (PHY)",
#           check_permutation_symmetries_complex_orbitals(onebody_matrix,
#                                                         phy_tbt))
#
#     print("Check permutation Symmetry After correction (PHY)",
#           check_permutation_symmetries_complex_orbitals(
#               onebody_matrix, phy_tbt))
#
#     new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
#         spinorbitals_to_orbitals(onebody_matrix, phy_tbt))
#
#     print("Spin symmetry broken (PHY)", spin_symm_broken)
#     result = single_qchem_dmrg_calc(
#         new_one_body_tensor, new_two_body_tensor, dmrg_param)
#     return result
