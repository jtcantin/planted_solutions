import CAS_Cropping.ferm_utils as feru
import CAS_Cropping.csa_utils as csau
import CAS_Cropping.var_utils as varu
from CAS.dmrghandler.src.dmrghandler.qchem_dmrg_calc import single_qchem_dmrg_calc
from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import check_spin_symmetry, spinorbitals_to_orbitals
from DMRG_simulation.validator.symmetry_check import check_permutation_symmetries_complex_orbitals
from DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation
import numpy as np

def get_dmrg_energy(Hf, spin_orb, dmrg_param):
    """
    Use DMRG to calculate the ground state and ground state energy
    Args:
        Hf: FermionOperator representation of the Hamiltonian
        spin_orb: The number of spin orbitals
        dmrg_param: DMRG parameters

    Returns:

    """

    # Htbt is in a^ a a^ a format
    # Htbt = feru.get_chemist_tbt(Hf, spin_orb, spin_orb=True)
    Htbt = Hf

    # Get one body difference
    # one_body = varu.get_one_body_correction_from_tbt(Hf,
    #                                                  feru.get_chemist_tbt(Hf))
    #
    # # Get the one body tensor from the difference
    # onebody_matrix = feru.get_obt(one_body, n=spin_orb, spin_orb=True)
    onebody_matrix = np.zeros((spin_orb, spin_orb))
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
