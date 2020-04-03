#!/usr/bin/env python
"""
a single script for folding proteins with trRosetta + pyrosetta
if you publish a model made by this please cite:
J Yang et al, Improved protein structure prediction using predicted inter-residue orientations, PNAS (2020).
"""

import copy
import os
import pathlib
import numpy as np
import random
from typing import Dict, Any

from pyrosetta import (
    rosetta,
    MoveMap,
    SwitchResidueTypeSetMover,
    ScoreFunction,
    init,
    create_score_function,
    RepeatMover,
    pose_from_sequence,
)
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
import argparse


# These are program defaults
PARAMS = {
    "PCUT": 0.05,
    "PCUT1": 0.5,
    "EBASE": -0.5,
    "EREP": [10.0, 3.0, 0.5],
    "DREP": [0.0, 2.0, 3.5],
    "PREP": 0.1,
    "SIGD": 10.0,
    "SIGM": 1.0,
    "MEFF": 0.0001,
    "DCUT": 19.5,
    "ALPHA": 1.57,
    "DSTEP": 0.5,
    "ASTEP": 15.0,
}


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--npz", type=str, help="input distograms and anglegrams (NN predictions)", required=True)
    parser.add_argument("--fasta", type=str, help="input sequence as a fasta file", required=True)
    parser.add_argument("--output_pdb", type=str, help="output model (in PDB format)", required=True)
    parser.add_argument("--weights_dir", help="Directory containing scorefunction definitions", required=True)

    parser.add_argument(
        "-p", "--pcut", type=float, default=PARAMS["PCUT"], help="min probability of distance restraints"
    )
    parser.add_argument(
        "-m", "--mode", type=int, default=2, choices=[0, 1, 2], help="0: sh+m+l, 1: (sh+m)+l, 2: (sh+m+l)"
    )
    parser.add_argument("--workdir", type=str, default="workdir", help="folder to store temp files")
    parser.add_argument("-n", type=int, dest="steps", default=1000, help="number of minimization steps")
    parser.add_argument("--no-orient", dest="use_orient", action="store_false", default=True)
    parser.add_argument("--no-fastrelax", dest="fastrelax", action="store_false", default=True)
    args = parser.parse_args()
    return args


def gen_rst(npz: Dict[str, Any], tmpdir: str, params: Dict[str, Any]) -> Dict[str, Any]:
    dist, omega, theta, phi = npz["dist"], npz["omega"], npz["theta"], npz["phi"]

    # dictionary to store Rosetta restraints
    rst = {"dist": [], "omega": [], "theta": [], "phi": [], "rep": []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT = 0.05  # params['PCUT']
    EBASE = params["EBASE"]
    EREP = params["EREP"]
    DREP = params["DREP"]
    MEFF = params["MEFF"]
    DCUT = params["DCUT"]
    ALPHA = params["ALPHA"]

    DSTEP = params["DSTEP"]
    ASTEP = np.deg2rad(params["ASTEP"])

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25 + DSTEP * i for i in range(32)])
    prob = np.sum(dist[:, :, 5:], axis=-1)
    bkgr = np.array((bins / DCUT) ** ALPHA)
    attr = -np.log((dist[:, :, 5:] + MEFF) / (dist[:, :, -1][:, :, None] * bkgr[None, None, :])) + EBASE
    repul = np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None] + np.array(EREP)[None, None, :]
    dist = np.concatenate([repul, attr], axis=-1)
    bins = np.concatenate([DREP, bins])
    rosetta_spline_bins = rosetta.utility.vector1_double()
    for bin_ in bins:
        rosetta_spline_bins.append(bin_)
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    nbins = 35
    step = 0.5
    for a, b, p in zip(i, j, prob):
        if b > a:
            rosetta_spline_potential = rosetta.utility.vector1_double()
            for pot in dist[a, b]:
                rosetta_spline_potential.append(pot)
            spline = rosetta.core.scoring.func.SplineFunc(
                "", 1.0, 0.0, step, rosetta_spline_bins, rosetta_spline_potential
            )
            atom_id_a = rosetta.core.id.AtomID(5, a + 1)
            atom_id_b = rosetta.core.id.AtomID(5, b + 1)  # 5 is CB
            rst["dist"].append(
                [a, b, p, rosetta.core.scoring.constraints.AtomPairConstraint(atom_id_a, atom_id_b, spline)]
            )
    print(f"dist restraints: {len(rst['dist'])}")

    ########################################################
    # omega: -pi..pi
    ########################################################
    nbins = omega.shape[2] - 1 + 4
    bins = np.linspace(-np.pi - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
    rosetta_spline_bins = rosetta.utility.vector1_double()
    for bin_ in bins:
        rosetta_spline_bins.append(bin_)
    prob = np.sum(omega[:, :, 1:], axis=-1)
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    omega = -np.log((omega + MEFF) / (omega[:, :, -1] + MEFF)[:, :, None])
    omega = np.concatenate([omega[:, :, -2:], omega[:, :, 1:], omega[:, :, 1:3]], axis=-1)
    for a, b, p in zip(i, j, prob):
        if b > a:
            rosetta_spline_potential = rosetta.utility.vector1_double()
            for pot in omega[a, b]:
                rosetta_spline_potential.append(pot)
            spline = rosetta.core.scoring.func.SplineFunc(
                "", 1.0, 0.0, ASTEP, rosetta_spline_bins, rosetta_spline_potential
            )
            atom_id_1 = rosetta.core.id.AtomID(2, a + 1)  # CA
            atom_id_2 = rosetta.core.id.AtomID(5, a + 1)  # CB
            atom_id_3 = rosetta.core.id.AtomID(5, b + 1)
            atom_id_4 = rosetta.core.id.AtomID(2, b + 1)
            constraint = rosetta.core.scoring.constraints.DihedralConstraint(
                atom_id_1, atom_id_2, atom_id_3, atom_id_4, spline
            )
            rst["omega"].append([a, b, p, constraint])
    print(f"omega restraints: {len(rst['omega'])}")

    ########################################################
    # theta: -pi..pi
    ########################################################
    # use bins from omega
    prob = np.sum(theta[:, :, 1:], axis=-1)
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    theta = -np.log((theta + MEFF) / (theta[:, :, -1] + MEFF)[:, :, None])
    theta = np.concatenate([theta[:, :, -2:], theta[:, :, 1:], theta[:, :, 1:3]], axis=-1)
    for a, b, p in zip(i, j, prob):
        if b != a:
            rosetta_spline_potential = rosetta.utility.vector1_double()
            for pot in theta[a, b]:
                rosetta_spline_potential.append(pot)

            spline = rosetta.core.scoring.func.SplineFunc(
                "", 1.0, 0.0, ASTEP, rosetta_spline_bins, rosetta_spline_potential
            )
            atom_id_1 = rosetta.core.id.AtomID(1, a + 1)  # N
            atom_id_2 = rosetta.core.id.AtomID(2, a + 1)  # CA
            atom_id_3 = rosetta.core.id.AtomID(5, a + 1)  # CB
            atom_id_4 = rosetta.core.id.AtomID(5, b + 1)
            constraint = rosetta.core.scoring.constraints.DihedralConstraint(
                atom_id_1, atom_id_2, atom_id_3, atom_id_4, spline
            )
            rst["theta"].append([a, b, p, constraint])
    print(f"theta restraints: {len(rst['theta'])}")

    ########################################################
    # phi: 0..pi
    ########################################################
    nbins = phi.shape[2] - 1 + 4
    bins = np.linspace(-1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
    rosetta_spline_bins = rosetta.utility.vector1_double()
    for bin_ in bins:
        rosetta_spline_bins.append(bin_)
    prob = np.sum(phi[:, :, 1:], axis=-1)
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    phi = -np.log((phi + MEFF) / (phi[:, :, -1] + MEFF)[:, :, None])
    phi = np.concatenate([np.flip(phi[:, :, 1:3], axis=-1), phi[:, :, 1:], np.flip(phi[:, :, -2:], axis=-1)], axis=-1)
    for a, b, p in zip(i, j, prob):
        if b != a:
            rosetta_spline_potential = rosetta.utility.vector1_double()
            for pot in phi[a, b]:
                rosetta_spline_potential.append(pot)
            spline = rosetta.core.scoring.func.SplineFunc(
                "", 1.0, 0.0, ASTEP, rosetta_spline_bins, rosetta_spline_potential
            )
            atom_id_1 = rosetta.core.id.AtomID(2, a + 1)  # CA
            atom_id_2 = rosetta.core.id.AtomID(5, a + 1)  # CB
            atom_id_3 = rosetta.core.id.AtomID(5, b + 1)
            constraint = rosetta.core.scoring.constraints.AngleConstraint(atom_id_1, atom_id_2, atom_id_3, spline)
            rst["phi"].append([a, b, p, constraint])

    print(f"phi restraints: {len(rst['phi'])}")
    return rst


def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
        phi, psi = random_dihedral()
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, 180)
    return pose


# pick phi/psi randomly from:
# -140  153 180 0.135 B
# -72  145 180 0.155 B
# -122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi = 0
    psi = 0
    r = random.random()
    if r <= 0.135:
        phi = -140
        psi = 153
    elif r > 0.135 and r <= 0.29:
        phi = -72
        psi = 145
    elif r > 0.29 and r <= 0.363:
        phi = -122
        psi = 117
    elif r > 0.363 and r <= 0.485:
        phi = -82
        psi = -14
    elif r > 0.485 and r <= 0.982:
        phi = -61
        psi = -41
    else:
        phi = 57
        psi = 39
    return (phi, psi)


def read_fasta(filename: str):
    fasta = ""
    with open(filename) as fh:
        for line in fh:
            if line[0] == ">":
                continue
            else:
                line = line.strip()
                fasta = fasta + line
    return fasta


def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


def add_rst(pose, rst: Dict[str, Any], sep1: int, sep2: int, params: Dict[str, Any], nogly: bool = False) -> None:

    pcut = params["PCUT"]
    seq = params["seq"]

    array = []

    if nogly:
        array += [
            rosetta_cst
            for a, b, p, rosetta_cst in rst["dist"]
            if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != "G" and seq[b] != "G" and p >= pcut
        ]
        if params["USE_ORIENT"]:
            array += [
                rosetta_cst
                for a, b, p, rosetta_cst in rst["omega"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != "G" and seq[b] != "G" and p >= pcut + 0.5
            ]  # 0.5
            array += [
                rosetta_cst
                for a, b, p, rosetta_cst in rst["theta"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != "G" and seq[b] != "G" and p >= pcut + 0.5
            ]  # 0.5
            array += [
                rosetta_cst
                for a, b, p, rosetta_cst in rst["phi"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != "G" and seq[b] != "G" and p >= pcut + 0.6
            ]  # 0.6
    else:
        array += [
            rosetta_cst
            for a, b, p, rosetta_cst in rst["dist"]
            if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut
        ]
        if params["USE_ORIENT"]:
            array += [
                rosetta_cst
                for a, b, p, rosetta_cst in rst["omega"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5
            ]
            array += [
                rosetta_cst
                for a, b, p, rosetta_cst in rst["theta"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5
            ]
            array += [
                rosetta_cst
                for a, b, p, rosetta_cst in rst["phi"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.6
            ]  # 0.6

    if len(array) < 1:
        return

    random.shuffle(array)

    constraints = rosetta.core.scoring.constraints.ConstraintSet()
    for constraint in array:
        constraints.add_constraints(constraint)

    # add to pose
    csm = rosetta.protocols.constraint_movers.ConstraintSetMover()
    csm.constraint_set(constraints)
    csm.add_constraints(True)
    csm.apply(pose)


def main(args):
    params = copy.deepcopy(PARAMS)
    params["USE_ORIENT"] = args.use_orient
    params["PCUT"] = args.pcut
    workdir = args.workdir

    # init PyRosetta
    init("-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100")

    # Create temp folder to store all the restraints
    pathlib.Path(workdir).mkdir(exist_ok=True, parents=True)

    # read and process restraints & sequence
    npz = np.load(args.npz)
    seq = read_fasta(args.fasta)
    rst = gen_rst(npz, workdir, params)
    params["seq"] = seq

    ########################################################
    # Scoring functions and movers
    ########################################################
    sf = ScoreFunction()
    sf_base_fn = os.path.join(args.weights_dir, "scorefxn.wts")
    sf.add_weights_from_file(sf_base_fn)

    sf1 = ScoreFunction()
    sf_1_fn = os.path.join(args.weights_dir, "scorefxn1.wts")
    sf1.add_weights_from_file(sf_1_fn)

    sf_vdw = ScoreFunction()
    sf_vdw_fn = os.path.join(args.weights_dir, "scorefxn_vdw.wts")
    sf_vdw.add_weights_from_file(sf_vdw_fn)

    sf_cart = ScoreFunction()
    sf_cart_fn = os.path.join(args.weights_dir, "scorefxn_cart.wts")
    sf_cart.add_weights_from_file(sf_cart_fn)

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover = MinMover(mmap, sf, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover.max_iter(1000)

    min_mover1 = MinMover(mmap, sf1, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover1.max_iter(1000)

    min_mover_vdw = MinMover(mmap, sf_vdw, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover_vdw.max_iter(500)

    min_mover_cart = MinMover(mmap, sf_cart, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover_cart.max_iter(1000)
    min_mover_cart.cartesian(True)

    repeat_mover = RepeatMover(min_mover, 3)

    ########################################################
    # initialize pose
    ########################################################
    pose = pose_from_sequence(seq, "centroid")

    # mutate GLY to ALA
    for i, a in enumerate(seq):
        if a == "G":
            mutator = rosetta.protocols.simple_moves.MutateResidue(i + 1, "ALA")
            mutator.apply(pose)
            print(f"mutation: G{i+1}A")

    set_random_dihedral(pose)
    remove_clash(sf_vdw, min_mover_vdw, pose)

    ########################################################
    # minimization
    ########################################################

    if args.mode == 0:

        # short
        print("short")
        add_rst(pose, rst, 1, 12, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

        # medium
        print("medium")
        add_rst(pose, rst, 12, 24, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

        # long
        print("long")
        add_rst(pose, rst, 24, len(seq), params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

    elif args.mode == 1:

        # short + medium
        print("short + medium")
        add_rst(pose, rst, 3, 24, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

        # long
        print("long")
        add_rst(pose, rst, 24, len(seq), params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

    elif args.mode == 2:

        # short + medium + long
        print("short + medium + long")
        add_rst(pose, rst, 1, len(seq), params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

    # mutate ALA back to GLY
    for i, a in enumerate(seq):
        if a == "G":
            mutator = rosetta.protocols.simple_moves.MutateResidue(i + 1, "GLY")
            mutator.apply(pose)
            print(f"mutation: G{i+1}A")

    ########################################################
    # full-atom refinement
    ########################################################

    if args.fastrelax:
        print("running fast relax")

        sf_fa = create_score_function("ref2015")
        sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
        sf_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
        sf_fa.set_weight(rosetta.core.scoring.angle_constraint, 1)

        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)

        relax = rosetta.protocols.relax.FastRelax()
        relax.set_scorefxn(sf_fa)
        relax.max_iter(200)
        relax.dualspace(True)
        relax.set_movemap(mmap)

        pose.remove_constraints()
        switch = SwitchResidueTypeSetMover("fa_standard")
        switch.apply(pose)

        print("relax...")
        params["PCUT"] = 0.15
        add_rst(pose, rst, 1, len(seq), params, True)
        relax.apply(pose)

    ########################################################
    # save final model
    ########################################################
    pose.dump_pdb(args.output_pdb)


if __name__ == "__main__":
    main(parseargs())
