from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import load_tensors_from_fcidump
from CAS.dmrghandler.src.dmrghandler.pyscf_wrappers import two_body_tensor_orbital_to_spin_orbital, one_body_tensor_orbital_to_spin_orbital
import numpy as np
import CAS_Cropping.ferm_utils as feru
from DMRG_simulation.format_dmrg.format_tensor import physicist_to_chemist
from DMRG_simulation.utils.fci_ground_energy import (get_fci_ground_energy,
                                                     get_ground_state_manually)
from DMRG_simulation.format_dmrg.format_dmrg_param import (
    get_dmrg_param, get_dmrg_process_param)

from DMRG_simulation.dmrg_simulator.dmrg_chem_tbt import get_dmrg_from_chem, get_dmrg_from_chem_original
from DMRG_simulation.dmrg_simulator.dmrg_phy_tensor import get_dmrg_from_phy
from pyscf.fci.direct_nosym import absorb_h1e
from pyscf import ao2mo
from CAS.dmrghandler.src.dmrghandler.qchem_dmrg_calc import single_qchem_dmrg_calc
from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import (
    check_spin_symmetry, spinorbitals_to_orbitals)
import openfermion as of
from DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation

default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8


def load_tensor(path_to_data):
    """
    Loading tensors from FCIDump file.
    Args:
        path_to_data: Path to the catalysts FCIDump file.

    Returns:

    """
    load_result = {}
    (
        one_body_tensor,
        two_body_tensor,
        nuc_rep_energy,
        num_orbitals,
        num_spin_orbitals,
        num_electrons,
        two_S,
        two_Sz,
        orb_sym,
        extra_attributes,
    ) = load_tensors_from_fcidump(path_to_data)

    load_result['one_body_tensor'] = one_body_tensor_orbital_to_spin_orbital(one_body_tensor)
    load_result['two_body_tensor'] = two_body_tensor_orbital_to_spin_orbital(two_body_tensor)
    load_result['nuc_rep_energy'] = nuc_rep_energy
    load_result['num_orbitals'] = num_orbitals
    load_result['num_spin_orbitals'] = num_spin_orbitals
    load_result['num_electrons'] = num_electrons
    load_result['two_S'] = two_S
    load_result['two_Sz'] = two_Sz
    load_result['orb_sym'] = orb_sym
    load_result['extra_attributes'] = extra_attributes

    absorbed_tbt = absorb_h1e(one_body_tensor, two_body_tensor, norb=num_orbitals,
                              nelec=num_electrons)
    htbt_added_phy = ao2mo.restore(1, absorbed_tbt.copy(),
                                   num_orbitals).astype(one_body_tensor.dtype,copy=False)
    load_result['combined_chem_tbt'] = two_body_tensor_orbital_to_spin_orbital(0.5 * htbt_added_phy)
    return load_result


if __name__ == '__main__':
    result = load_tensor("../data/fcidump.2_co2_6-311++G**")
    # result = load_tensor("../data/fcidump.2_co2_6-311++G**planted_original")
    # result = load_tensor("../data/fcidump.7_melact_6-311++G**")

    # Everything in spin orbital
    htbt_here = result['combined_chem_tbt']
    obt_phy = result['one_body_tensor']
    tbt_phy = result['two_body_tensor']
    spin_orbs = result["num_spin_orbitals"]
    num_elec = result['num_electrons']
    zero_matrix = np.zeros((spin_orbs, spin_orbs))


    one_body, two_body = physicist_to_chemist(obt_phy, tbt_phy,
                                              result["num_spin_orbitals"])

    # one_body, two_body = physicist_to_chemist(np.zeros((spin_orbs, spin_orbs)),
    #                                           combined_physicist,
    #                                           result["num_spin_orbitals"])
    print(spin_orbs)
    print(result["two_body_tensor"].shape)
    print(htbt_here.shape)
    print(two_body.shape)
    onebody_tbt = feru.onebody_to_twobody(one_body)
    # htbt_here is in chemist notation (One body two body combined)
    htbt_here_new = np.add(two_body, onebody_tbt)
    tbt = feru.get_ferm_op(two_body, spin_orb=True)
    tbt.compress(abs_tol=0.1)


    k = [[i for i in range(result["num_spin_orbitals"])]]
    e_nums = [result["num_electrons"]]
    print(e_nums)

    num_orbitals = result["num_spin_orbitals"] // 2
    num_electrons = result["num_electrons"]
    num_spin_orbitals = result["num_spin_orbitals"]
    num_unpaired_electrons = result["two_S"]
    multiplicity = result["two_Sz"] * 2 + 1

    init_state_bond_dimension = 100
    max_num_sweeps = 200
    energy_convergence_threshold = 1e-8
    sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
    sweep_schedule_noise = default_sweep_schedule_noise
    sweep_schedule_davidson_threshold = (
        default_sweep_schedule_davidson_threshold
    )
    dmrg_process_param = get_dmrg_process_param(
        init_state_bond_dimension, max_num_sweeps,
        energy_convergence_threshold,
        sweep_schedule_bond_dims, sweep_schedule_noise,
        sweep_schedule_davidson_threshold)

    dmrg_param = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons,
        multiplicity, dmrg_process_param)

    num_unpaired_electrons2 = 0
    multiplicity2 = 1

    dmrg_param2 = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons2,
        multiplicity2, dmrg_process_param)

    Hf = feru.get_ferm_op_two(htbt_here_new, spin_orb=True)
    ground_energy_tbt = get_fci_ground_energy(Hf, spin_orbs)
    # print("Ground energy manually found:", get_ground_state_manually(htbt_here, k, e_nums)[0])
    print("FCI result with CHEM obt & tbt:", ground_energy_tbt[0])

    chemist_original, chem_obt, chem_tbt = get_dmrg_from_chem_original(one_body, two_body, result["num_spin_orbitals"], dmrg_param)
    # print("DMRG with physicist combined:", get_dmrg_from_phy(np.zeros((spin_orbs, spin_orbs)), combined_physicist, dmrg_param)['dmrg_ground_state_energy'])
    # DMRG result is calculated using spatial orbitals
    print("DMRG result with PHY obt & tbt:", get_dmrg_from_phy(result["one_body_tensor"], result["two_body_tensor"], dmrg_param)["dmrg_ground_state_energy"])
    print("DMRG result with CHEM obt & tbt:", chemist_original)
    print("DMRG result with CHEM combined:", get_dmrg_from_chem(htbt_here, result["num_spin_orbitals"], dmrg_param))
    print("tbt shape", result["two_body_tensor"].shape)
    print("# of spin orbs", result["num_spin_orbitals"])
    print("# of electrons", result["num_electrons"])
    print("2S", result["two_S"])
    print("Sz", result["two_Sz"])


