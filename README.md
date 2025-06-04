# Planted solutions
Generation of cropped complete active space (CCAS) planted solution Hamiltonians. These planted solutions are based on extant one- and two-electron integrals that are cropped to be "block-diagonal". Balance operators are added to enforce a specific number of electrons in each block. Randomized killer operators and orbital rotations are then included to obfuscate the structure of the Hamiltonian.

The ground state energy of each planted solution is obtained by summing the exact ground state energies of each non-obfuscated block. The obfuscating operators (killer and orbital rotations) do not modify the ground state energy.

# Instructions
To run:

1. Download the github repository
2. Update the parameters in `planted_hamiltonian_generator.ipynb`
3. Put fcidump files (or gzipped versions) into the folder designated by `fcidump_path` 
4. Run the `planted_hamiltonian_generator.ipynb` notebook

Four files per input fcidump file will be generated:
1. planted solution fcidump
2. compressed planted solution fcidump
3. *performer* problem instance json file
4. *proctor* problem instance json file

The *performer* problem instance json file contains the information needed for a performer to obtain the planted solution fcidump file (assuming the proctor has placed the file appropriately) as well as information about any requirements (e.g., max run time). This file does not contain information that could be used by a performer to know the underlying Hamiltonian strucure and thus potentially "cheat".

The *proctor* problem instance json file contains the information as in the performer file, plus information required to reproduce the planted solution fcidump file. 

# Authors
This code is composed of or based on code written by (alphabetical order) Joshua T. Cantin [jtcantin], Rick Huang [Rick0317], Ignacio Loaiza [iloaiza], Luis A. Martínez-Martínez [lamq317], Smik Patel [SmikPatel] Linjun Wang [Zephrous5747], and Tzu-Ching (Thomson) Yen [ThomsonYen].

# Article
Article to be cited.

# Ground State Energy Estimation Benchmark
Proctor problem instance json files for the [GSEE benchmark](https://github.com/isi-usc-edu/qb-gsee-benchmark) can be found in the folder `proctor_jsons`.
