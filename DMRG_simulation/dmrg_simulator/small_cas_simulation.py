import openfermion as of
import numpy as np
import CAS.saveload_utils as sl
from DMRG_simulation.dmrg_simulation import get_ground_state_fci
from DMRG_simulation.format_dmrg.format_dmrg_param import get_dmrg_param, get_dmrg_process_param
from DMRG_simulation.dmrg_simulator.dmrg_chem_tbt import get_dmrg_from_chem, get_dmrg_from_chem_original
import CAS.saveload_utils as sl
import CAS.ferm_utils as feru
import CAS.var_utils as varu
import sys
sys.path.append("../../CAS/")

default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8


if __name__ == '__main__':
    mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
    ground_energy, ground_state = get_ground_state_fci(mol, path="../../CAS/")
    Hf = sl.load_fermionic_hamiltonian(mol, prefix="../../CAS/")
    print("Hamiltonian in physicist", Hf)
    spin_orb = of.count_qubits(Hf)
    two_body = feru.get_chemist_tbt(Hf, spin_orb, spin_orb=True)
    print("before adding:", feru.get_ferm_op(two_body, spin_orb=True))
    one_body = varu.get_one_body_correction_from_tbt(Hf,
                                                     feru.get_chemist_tbt(Hf))
    onebody_matrix = feru.get_obt(one_body, n=spin_orb, spin_orb=True)
    print("one_body_term:", one_body)
    onebody_tbt = feru.onebody_to_twobody(onebody_matrix)

    # Everything in two body tensor
    Htbt = np.add(two_body, onebody_tbt)
    htbt_in_ferm = feru.get_ferm_op(Htbt, spin_orb=True)
    htbt_in_ferm.compress(abs_tol=0.09)
    print("Added:", htbt_in_ferm)

    num_orbitals = spin_orb // 2
    num_electrons = 2
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

    # Hf = feru.get_ferm_op(htbt_here, spin_orb=True)
    # ground_energy_tbt = get_fci_ground_energy(Hf)
    # print("Ground energy manually found:", get_ground_state_manually(htbt_here, k, e_nums)[0])
    # print("Ground energy found:", ground_energy_tbt[0])

    chemist_original, chem_obt, chem_tbt = get_dmrg_from_chem_original(onebody_matrix, two_body, spin_orb, dmrg_param)

    # print("DMRG result in PHY:", get_dmrg_from_phy(result["one_body_tensor"], result["two_body_tensor"], dmrg_param)["dmrg_ground_state_energy"])
    print("DMRG with chemist original", chemist_original)
    print("DMRG with chemist:", get_dmrg_from_chem(Htbt, spin_orb, dmrg_param))
