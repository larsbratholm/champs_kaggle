"""
Create the Kaggle data from Gaussian output files
"""

import pybel
import numpy as np
import openbabel
import os
import tarfile

def mol_read_conn(mol):
    """
    Get connectivity of molecule
    """
    atoms = len(mol.atoms)
    conn_mat = np.zeros((atoms, atoms), dtype=int)
    for a1 in range(atoms):
        for nbr_atom in openbabel.OBAtomAtomIter(mol.atoms[a1].OBAtom):
            a2 = nbr_atom.GetId()
            bond = mol.atoms[a1].OBAtom.GetBond(nbr_atom)
            conn_mat[a1][a2] = 1
            conn_mat[a2][a1] = 1

    return conn_mat

def conn_distance(conn_mat):
    """
    Get connectivity distance between atoms.
    Floyd-Warshall algorithm
    """
    n_atoms = conn_mat.shape[0]

    distance = conn_mat.copy()
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if distance[i,j] == 0:
                distance[i,j] = n_atoms
                distance[j,i] = n_atoms

    for k in range(n_atoms):
        for i in range(n_atoms):
            for j in range(i+1,n_atoms):
                distance[i,j] = min(distance[i,j], distance[i,k] + distance[k,j])
                distance[j,i] = distance[i,j]

    return distance

def jread(lines, atomnumber):
    """
    Parse the sum of all j-coupling contributions (total).
    """

    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    skip = True
    for line in lines:
        if "Total nuclear spin-spin coupling J (Hz):" in line:
            skip = False
            continue
        elif "End of Minotr" in line:
            break
        elif skip:
            continue

        if "D" not in line:
            tokens = line.split()
            i_indices = np.asarray(tokens, dtype=int)
        else:
            tokens = line.split()
            index_j = int(tokens[0]) - 1
            for i in range(len(tokens)-1):
                index_i = i_indices[i] - 1
                coupling = float(tokens[i+1].replace("D","E"))
                couplings[index_i, index_j] = coupling
                couplings[index_j, index_i] = coupling
    return couplings

def read_dipole_moment(lines, basename):
    dipole_moment = None
    skip_line = False
    for line in lines:
        if line.startswith(" Dipole moment"):
            skip_line = True
        elif skip_line:
            dipole_moment = np.asarray(line.split()[1:6:2], dtype=float)
            break
    if dipole_moment is None or dipole_moment.size != 3:
        raise SystemExit("Error reading dipole moment in %s" % basename)
    return dipole_moment

def read_potential_energy(lines, basename):
    potential_energy = None
    for line in lines:
        if line.startswith(" SCF Done:"):
            potential_energy = float(line.split()[4])
            break
    if potential_energy is None:
        raise SystemExit("Error reading potential energy in %s" % basename)
    return potential_energy

def read_magnetic_shielding_tensor(lines, basename):
    shieldings = []
    atom_shieldings = []
    skip = True
    for line in lines:
        if line.startswith(" SCF GIAO"):
            skip = False
            continue
        elif skip:
            continue
        elif line.startswith(" There are"):
            break
        elif "Isotropic" in line:
            continue
        elif line.startswith("   Eigenvalues"):
            shieldings.append(np.asarray(atom_shieldings, dtype=float))
            atom_shieldings = []
            continue
        tokens = line.split()
        atom_shieldings.extend(tokens[1::2])
    shieldings_array = np.asarray(shieldings, dtype=float)
    if shieldings_array.shape[1] != 9:
        raise SystemExit("Error reading magnetic shielding tensor in %s" % basename)
    return shieldings_array

def read_mulliken_charges(lines, basename):
    mulliken_charges = []
    c = -1
    for line in lines:
        if line.startswith(" Mulliken charges:") or line.startswith(" Mulliken atomic charges:"):
            c = 1
        elif line.startswith(" Sum of Mulliken charges") or line.startswith(" Sum of Mulliken atomic charges"):
            break
        elif c == 1:
            c = 0
        elif c == 0:
            tokens = line.split()
            mulliken_charges.append(float(tokens[2]))

    if len(mulliken_charges) == 0:
        raise SystemExit("Error reading mulliken charges in %s" % basename)
    return np.asarray(mulliken_charges, dtype=float)

def jread_fc(lines, atomnumber):
    """
    Parse the FC contribution to j-couplings
    """
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    skip = True
    for line in lines:
        if line.startswith(" Fermi Contact (FC) contribution to J (Hz):"):
            skip = False
            continue
        elif line.startswith(" Spin-dipolar (SD) contribution to K (Hz):"):
            break
        elif skip:
            continue

        if "D" not in line:
            tokens = line.split()
            i_indices = np.asarray(tokens, dtype=int)
        else:
            tokens = line.split()
            index_j = int(tokens[0]) - 1
            for i in range(len(tokens)-1):
                index_i = i_indices[i] - 1
                coupling = float(tokens[i+1].replace("D","E"))
                couplings[index_i, index_j] = coupling
                couplings[index_j, index_i] = coupling
    return couplings

def jread_sd(lines, atomnumber):
    """
    Parse the SD contribution to j-couplings
    """
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    skip = True
    for line in lines:
        if line.startswith(" Spin-dipolar (SD) contribution to J (Hz):"):
            skip = False
            continue
        elif line.startswith(" Paramagnetic spin-orbit (PSO) contribution to K (Hz):"):
            break
        elif skip:
            continue

        if "D" not in line:
            tokens = line.split()
            i_indices = np.asarray(tokens, dtype=int)
        else:
            tokens = line.split()
            index_j = int(tokens[0]) - 1
            for i in range(len(tokens)-1):
                index_i = i_indices[i] - 1
                coupling = float(tokens[i+1].replace("D","E"))
                couplings[index_i, index_j] = coupling
                couplings[index_j, index_i] = coupling
    return couplings

def jread_pso(lines, atomnumber):
    """
    Parses the PSO contribution to the j-couplings
    """
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    skip = True
    for line in lines:
        if line.startswith(" Paramagnetic spin-orbit (PSO) contribution to J (Hz):"):
            skip = False
            continue
        elif line.startswith(" Diamagnetic spin-orbit (DSO) contribution to K (Hz):"):
            break
        elif skip:
            continue

        if "D" not in line:
            tokens = line.split()
            i_indices = np.asarray(tokens, dtype=int)
        else:
            tokens = line.split()
            index_j = int(tokens[0]) - 1
            for i in range(len(tokens)-1):
                index_i = i_indices[i] - 1
                coupling = float(tokens[i+1].replace("D","E"))
                couplings[index_i, index_j] = coupling
                couplings[index_j, index_i] = coupling
    return couplings

def jread_dso(lines, atomnumber):
    """
    Parses the DSO contribution to the j-couplings.
    """
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    skip = True
    for line in lines:
        if line.startswith(" Diamagnetic spin-orbit (DSO) contribution to J (Hz):"):
            skip = False
            continue
        elif line.startswith(" Total nuclear spin-spin coupling K (Hz):"):
            break
        elif skip:
            continue

        if "D" not in line:
            tokens = line.split()
            i_indices = np.asarray(tokens, dtype=int)
        else:
            tokens = line.split()
            index_j = int(tokens[0]) - 1
            for i in range(len(tokens)-1):
                index_i = i_indices[i] - 1
                coupling = float(tokens[i+1].replace("D","E"))
                couplings[index_i, index_j] = coupling
                couplings[index_j, index_i] = coupling
    return couplings

def parse_data():
    """
    Parse the gaussian output files and xyz files.
    """

    data = []
    contrib = []
    pot = []
    shield = []
    mulliken = []
    dipole = []
    coords = []

    atomnumber_to_letter = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F"}

    #TODO fix hard link
    ignored_structures = np.concatenate([np.loadtxt('../molecules_without_hydrogens.txt', dtype=str),
                                         np.loadtxt('../molecules_with_outliers.txt', dtype=str)])

    # Get script location
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Read the tar file containing all the log files
    try:
        log_tar = tarfile.open(script_dir + "/gaussian_output_files.tar.gz", "r:gz")
    except FileNotFoundError:
        print("gaussian_output_files.tar.gz not found.")
        quit()
    # Read the tar file containing all the xyz files
    try:
        xyz_tar = tarfile.open(script_dir + "/../data/xyz_files.tar.gz", "r:gz")
    except FileNotFoundError:
        print("xyz_files.tar.gz not found.")
        quit()

    c = 0
    for member in log_tar.getmembers():
        c += 1
        if c >= 10:
            break
        if member.isfile():
            basename = member.name.split("/")[-1].split(".")[0]
            if basename in ignored_structures:
                print("ignored", basename)
                continue
            with log_tar.extractfile(member) as f:
                # tarfile doesn't decode the object, so just do it manually
                log_lines = [byte.decode('utf-8') for byte in f.readlines()]

            with xyz_tar.extractfile(basename + ".xyz") as f:
                # A bit of a roundabout way due to limitations in tarfile and pybel
                # tarfile doesn't decode the object, so just do it manually
                xyz_string = "".join([byte.decode('utf-8') for byte in f.readlines()])

            # Convert xyz string to pybel mol object
            mol = pybel.readstring('xyz', xyz_string)
            n_atoms = len(mol.atoms)

            # Read connectivity matrix
            conn_mat = mol_read_conn(mol)
            # Get number of bonds separating atom pairs
            conn_dist = conn_distance(conn_mat)

            atom_types = np.asarray([x.atomicnum for x in mol.atoms])

            # Read all the scalar coupling constants
            # and their decomposition
            j_array = jread(log_lines, n_atoms)
            j_array_fc = jread_fc(log_lines, n_atoms)
            j_array_sd = jread_sd(log_lines, n_atoms)
            j_array_pso = jread_pso(log_lines, n_atoms)
            j_array_dso = jread_dso(log_lines, n_atoms)

            # Only look at couplings between H and H, C or N.
            atom1_loc = np.where(atom_types == 1)[0]
            atom2_loc = np.where(np.isin(atom_types, [1, 6, 7]))[0]
            for i in atom1_loc:
                for j in atom2_loc:
                    atype = atomnumber_to_letter[atom_types[j]]
                    if atype == "H" and i >= j:
                        continue

                    if conn_dist[i,j] == 0 or conn_dist[i,j] > 3:
                        continue

                    ctype = str(conn_dist) + "JH" + atype

                    data.append([basename, i, j, ctype, j_array[i, j]])
                    contrib.append([basename, i, j, ctype, j_array_fc[i, j], j_array_sd[i, j], j_array_pso[i, j], j_array_dso[i, j]])

            #TODO remove
            if len(atom1_loc) == 0 or len(atom2_loc) == len(atom1_loc):
                print("No couplings in:", basename)
                continue

            potential_energy = read_potential_energy(log_lines, basename)
            pot.append([basename, potential_energy])

            shielding_tensor = read_magnetic_shielding_tensor(log_lines, basename)
            for i in range(n_atoms):
                shield.append([basename, i] + shielding_tensor[i].tolist())

            mulliken_charges = read_mulliken_charges(log_lines, basename)
            for i in range(n_atoms):
                mulliken.append([basename, i, mulliken_charges[i]])

            dipole_moment = read_dipole_moment(log_lines, basename)
            for i in range(n_atoms):
                dipole.append([basename, i] + dipole_moment.tolist())

            for i, atom in enumerate(mol):
                atype = atomnumber_to_letter[atom_types[j]]
                coords.append([basename, i, atype] + list(atom.coords))


    return data, contrib, pot, shield, mulliken, dipole, coords

def create_dataset():
    """
    Create the Kaggle dataset
    """
    data, contrib, pot, shield, mulliken, dipole, coords = parse_data()
    print(pot)
    #write_data()

def consistency_check(filename):
    with open(filename) as f:
        lines = f.readlines()

    inchi1, inchi2 = lines[-1].split()
    if inchi1 != inchi2:
        return False
    smiles1, smiles2 = lines[-2].split()
    if smiles1 != smiles2:
        return False
    return True

#TODO remove outliers etc.

if __name__ == "__main__":
    #xyz_folder = "/home/lb17101/dev/qm9_nmr/xyz/"
    #output_folder = "dataset/"
    create_dataset()


    #d = {}
    #atomnumber_to_name = {1:"H", 6:"C", 7:"N", 9:"F"}
    #empty = 0
    #for filename in filenames:
    #    output = make_jxyz_nmrlog(filename, "qm9")
    #    if sum(output) == 0:
    #        print(filename)
    #        empty += 1
    #        continue
    #    c = 0
    #    for i in (1,2,3):
    #        for j in (1,6,7,9):
    #            if i == 1 and j in [1,9]:
    #                continue
    #            key = "%dJ-H%s" % (i, atomnumber_to_name[j])
    #            if key not in d: d[key] = []
    #            d[key].append(output[c])
    #            c += 1

    #print(len(filenames))
    #all_ = np.zeros(len(filenames) - empty, dtype=int)
    #for key, value in d.items():
    #    print(key, np.mean(d[key]), np.min(d[key]), np.max(d[key]))
    #    all_ += np.asarray(value)
    #print("all", np.mean(all_), np.min(all_), np.max(all_))
