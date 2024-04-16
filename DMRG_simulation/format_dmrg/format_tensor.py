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
