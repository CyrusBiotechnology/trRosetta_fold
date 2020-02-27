# trRosetta_fold


A slightly modified version of the trRosetta folding protocol. 


original readme here
--------------------

Brief introduction for structure prediction by trRosetta

Step 0. Please generate an MSA (in a3m format) for your protein sequence from other softwares, such as HHblits (https://github.com/soedinglab/hh-suite).

Step 1. Using the generated MSA, predict the distance and orientations by running the scripts at: https://github.com/gjoni/trRosetta.

Step 2. Generate structure models from trRosetta (need to install PyRosetta3: http://www.pyrosetta.org/dow/pyrosetta3-download)  
>> python trRosetta.py T1008.npz T1008.fasta model.pdb

We suggest running step two for multiple times to generate multiple models and select the top models based on the energy scores, 
which are available at the end of the model's pdb file.

More details about trRosetta can be found from the following paper:
J Yang et al, Improved protein structure prediction using predicted inter-residue orientations, PNAS (2020).


Please contact Jianyi (yangjy@nankai.edu.cn) if you have any comments or problems.


Jianyi Yang
2019.08
