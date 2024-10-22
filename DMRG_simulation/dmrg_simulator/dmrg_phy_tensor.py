from CAS.dmrghandler.src.dmrghandler.qchem_dmrg_calc import (
    single_qchem_dmrg_calc)

from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import (
    check_spin_symmetry, spinorbitals_to_orbitals)

def get_dmrg_from_phy(obt, tbt, dmrg_param):
    """

    Args:
        obt:
        tbt:
        spin_orb:
        dmrg_param:

    Returns:

    """
    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = spinorbitals_to_orbitals(obt, tbt)
    result = single_qchem_dmrg_calc(
        new_one_body_tensor, new_two_body_tensor, dmrg_param)
    return result
