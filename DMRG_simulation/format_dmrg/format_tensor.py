import numpy as np


def get_correct_permutation(obt, two_body_tensor, spin_orb):
    """
    Convert the conventional chemistry permutation into block2 permutation
    Args:
        obt: one body tensor for the hamiltonain
        two_body_tensor: two body tensor for the hamiltonian
        spin_orb: the number of spin orbitals

    Returns: Transformed one body tensor and the two body tensor.

    """
    n = spin_orb
    one_body_tensor = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            total = 0
            for i in range(n):
                total += two_body_tensor[p, i, i, q]
            one_body_tensor[p, q] = total
    corrected = obt + one_body_tensor
    tbt = 2 * two_body_tensor

    return corrected, tbt


def physicist_to_chemist(obt, tbt, spin_orb):
    """
    Transform the physicist indices into chemist indices
    Args:
        obt:
        tbt:
        spin_orb:

    Returns:

    """
    n = spin_orb
    tbt_in_obt = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            total = 0
            for i in range(n):
                total += tbt[p, i, i, q]
            tbt_in_obt[p, q] = total

    one_body_tensor = np.subtract(obt, 0.5 * tbt_in_obt)
    two_body_tensor = 0.5 * tbt
    return one_body_tensor, two_body_tensor
