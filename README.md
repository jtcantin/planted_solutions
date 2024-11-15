# Planted solutions
Generation of planted solution Hamiltonians.

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
This code is composed of or based on code written by (alphabetical order) Joshua T. Cantin [jtcantin], Rick Huang [Rick0317], Ignacio Loaiza [iloaiza],   Linjun Wang [Zephrous5747], and Tzu-Ching (Thomson) Yen [ThomsonYen].

# Article
Article to be cited.

# Ground State Energy Estimation Benchmark
Proctor problem instance json files for the [GSEE benchmark](https://github.com/isi-usc-edu/qb-gsee-benchmark) can be found in the folder `proctor_jsons`.
