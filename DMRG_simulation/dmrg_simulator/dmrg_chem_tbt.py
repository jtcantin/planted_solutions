import CAS_Cropping.ferm_utils as feru
import CAS_Cropping.var_utils as varu
import openfermion as of
from CAS.dmrghandler.src.dmrghandler.qchem_dmrg_calc import single_qchem_dmrg_calc
from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import check_spin_symmetry, spinorbitals_to_orbitals
from DMRG_simulation.validator.symmetry_check import check_permutation_symmetries_complex_orbitals
import numpy as np
from DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation


def get_dmrg_from_chem(chem_tbt, spin_orb: int, dmrg_param: dict):
    """
    Get DMRG result from chemistry indices convention
    Args:
        chem_tbt:
        spin_orb: number of spin orbitals
        dmrg_param:

    Returns:

    """
    zero_matrix = np.zeros((spin_orb, spin_orb))
    # Get the physics indices convention
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(zero_matrix, chem_tbt, spin_orb))

    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

    result = single_qchem_dmrg_calc(
        new_one_body_tensor, new_two_body_tensor, dmrg_param)
    return result["dmrg_ground_state_energy"]


def get_dmrg_from_chem_original(chem_obt, chem_tbt, spin_orb: int, dmrg_param: dict):
    """
    Get DMRG result from chemistry indices convention
    Args:
        Hf: Hamiltonian in Fermion Operator
        spin_orb: number of spin orbitals

    Returns:

    """
    # Get the correct permutation of tensors for Block2
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(chem_obt, chem_tbt, spin_orb))

    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

    result = single_qchem_dmrg_calc(
        new_one_body_tensor, new_two_body_tensor, dmrg_param)
    return result["dmrg_ground_state_energy"], one_body_tensor, two_body_tensor
