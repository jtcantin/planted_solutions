import pickle
import CAS_Cropping.csa_utils as csau
from DMRG_simulation.utils.fci_ground_energy import get_fci_ground_energy
import numpy as np


def test_ground_energy_match(path, file_name, key = ""):
    with open(path + file_name, 'rb') as handle:
        dic = pickle.load(handle)
    if not key:
        key = list(dic.keys())[0]
    E_min = dic[key]["E_min"]
    cas_x = dic[key]["cas_x"]
    killer = dic[key]["killer"]
    k = dic[key]["k"]
    upnum = dic[key]["upnum"]
    spin_orbs = dic[key]["spin_orbs"]
    e_nums = dic[key]["e_nums"]
    sol = dic[key]["sol"]
    tbt = csau.get_cas_matrix(cas_x, spin_orbs, k)
    print("Loaded ground state energy:", E_min)
    print("FCI ground state energy from CAS hamiltonian", get_fci_ground_energy(tbt))
    assert np.isclose(get_fci_ground_energy(tbt), E_min, rtol=1e-5), \
        "Fci energy should match loaded energy"


if __name__ == "__main__":
    # sys.settrace(trace)
    ps_path = "../../CAS_Cropping/planted_solutions/"
    # File name in ps_path folder
    file_name = "2_co2_6-311++G___12_9d464efb-b312-45f8-b0ba-8c42663059dc.pkl"
    test_ground_energy_match(ps_path, file_name)
