import openfermion as of
import CAS.ferm_utils as feru

def get_fci_ground_energy(tbt):
    recombined = feru.get_ferm_op(tbt, True)
    sparse = of.linalg.get_sparse_operator(recombined)
    ground_energy, gs = of.linalg.get_ground_state(
        sparse, initial_guess=None
    )
    return ground_energy
