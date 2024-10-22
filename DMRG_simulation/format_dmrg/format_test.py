import openfermion as of
import CAS.saveload_utils as sl
import CAS.ferm_utils as feru
import CAS.var_utils as varu
from format_tensor import get_correct_permutation
import sys
sys.path.append("../../CAS/")
def get_mol(mol):
    # Load the hamiltonian
    Hf = sl.load_fermionic_hamiltonian(mol, prefix="../")
    spin_orb = of.count_qubits(Hf)

    # Htbt is in a^ a a^ a format
    Htbt = feru.get_chemist_tbt(Hf, spin_orb, spin_orb=True)

    phy_tbt = feru.get_two_body_tensor(Hf, spin_orb)

    print("Physics tbt", phy_tbt)
    # Get one body difference
    one_body = varu.get_one_body_correction_from_tbt(Hf,
                                                     feru.get_chemist_tbt(Hf))

    # Get the one body tensor from the difference
    onebody_matrix = feru.get_obt(one_body, n=spin_orb, spin_orb=True)
    obt, tbt = get_correct_permutation(onebody_matrix, Htbt, spin_orb)
    return onebody_matrix, Htbt, obt, tbt


if __name__ == '__main__':
    mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
    onebody_matrix, Htbt, obt, tbt = get_mol(mol)
    print("Before obt", onebody_matrix)
    print("After obt", obt)
    print("Before tbt", Htbt)
    print("After tbt", tbt)
