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

def parse_data(script_dir):
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

    ignored_structures = np.concatenate([np.loadtxt('molecules_without_hydrogens.txt', dtype=str),
                                         np.loadtxt('molecules_with_outliers.txt', dtype=str)])

    # Read the tar file containing all the log files
    try:
        log_tar = tarfile.open(script_dir + "/data/gaussian_output_files.tar.gz", "r:gz")
    except FileNotFoundError:
        print("gaussian_output_files.tar.gz not found.")
        quit()
    # Read the tar file containing all the xyz files
    try:
        xyz_tar = tarfile.open(script_dir + "/data/xyz_files.tar.gz", "r:gz")
    except FileNotFoundError:
        print("xyz_files.tar.gz not found.")
        quit()

    for member in log_tar.getmembers():
        if member.isfile():
            basename = member.name.split("/")[-1].split(".")[0]
            if basename in ignored_structures:
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

                    ctype = str(conn_dist[i,j]) + "JH" + atype

                    data.append(np.asarray([basename, i, j, ctype, "%.6f" % j_array[i, j]]))
                    contrib.append(np.asarray([basename, i, j, ctype, "%.6f" % j_array_fc[i, j], "%.6f" % j_array_sd[i, j], \
                            "%.6f" % j_array_pso[i, j], "%.6f" % j_array_dso[i, j]]))

            #TODO remove
            if len(atom1_loc) == 0 or (len(atom2_loc) == 1 and len(atom1_loc) == 1):
                print("No couplings in:", basename)
                continue

            # Parse remaining data types
            potential_energy = read_potential_energy(log_lines, basename)
            pot.append(np.asarray([basename, "%.7f" % potential_energy]))

            shielding_tensor = read_magnetic_shielding_tensor(log_lines, basename)
            for i in range(n_atoms):
                shield.append(np.asarray([basename, i] + ["%.4f" % value for value in shielding_tensor[i]]))

            mulliken_charges = read_mulliken_charges(log_lines, basename)
            for i in range(n_atoms):
                mulliken.append(np.asarray([basename, i, "%.6f" % mulliken_charges[i]]))

            dipole_moment = read_dipole_moment(log_lines, basename)
            dipole.append(np.asarray([basename] + ["%.4f" % value for value in dipole_moment]))

            for i, atom in enumerate(mol):
                atype = atomnumber_to_letter[atom_types[i]]
                coords.append(np.asarray([basename, i, atype] + ["%.9f" % value for value in atom.coords]))

    log_tar.close()
    xyz_tar.close()

    pot_array = np.asarray(pot)
    shield_array = np.asarray(shield)
    mulliken_array = np.asarray(mulliken)
    dipole_array = np.asarray(dipole)
    coords_array = np.asarray(coords)
    data_array = np.asarray(data)
    contrib_array = np.asarray(contrib)

    return data_array, contrib_array, pot_array, shield_array, mulliken_array, \
            dipole_array, coords_array

def sort_data(data, sort_idx):
    """
    Sort the data array according to the given order of indices
    """
    for idx in sort_idx:
        if "." in data[0,idx]:
            type_ = float
        else:
            try:
                int(data[0,idx])
                type_ = int
            except:
                type_ = str
        subarray = data[:,idx].astype(type_)
        # Mergesort preserves order
        order = np.argsort(subarray, kind='mergesort')
        data[:] = data[order]

def add_index_and_header(header, train, test, idx):
    """
    Add id column and header
    """

    train_index = np.empty((train.shape[0] + 1, train.shape[1] + int(idx)), dtype='<U32')
    test_index = np.empty((test.shape[0] + 1, test.shape[1] + int(idx)), dtype='<U32')

    train_index[0] = np.asarray(header.split(","), dtype='<U32')
    test_index[0] = train_index[0]

    if idx:
        train_index[1:,0] = np.arange(train.shape[0])
        test_index[1:,0] = np.arange(train.shape[0], train.shape[0] + test.shape[0])
        train_index[1:,1:] = train
        test_index[1:,1:] = test
    else:
        train_index[1:] = train
        test_index[1:] = test

    return train_index, test_index

def process_and_write(train_names, test_names, data, script_dir, basename, sort_idx, idx):
    """
    Splits a csv file into sorted and indexed train and test files.
    """

    if basename == 'data':
        header = "id,molecule_name,atom_index_0,atom_index_1,type,scalar_coupling_constant"
    elif basename == 'scalar_coupling_contributions':
        header = "molecule_name,atom_index_0,atom_index_1,type,fc,sd,pso,dso"
    elif basename == "potential_energy":
        header = "molecule_name,potential_energy"
    elif basename == "magnetic_shielding_tensors":
        header = "molecule_name,atom_index,XX,YX,ZX,XY,YY,ZY,XZ,YZ,ZZ"
    elif basename == "mulliken_charges":
        header = "molecule_name,atom_index,mulliken_charge"
    elif basename == "dipole_moments":
        header = "molecule_name,X,Y,Z"
    elif basename == "structures":
        header = "molecule_name,atom_index,atom,x,y,z"
    else:
        print("Unknown basename:", basename)
        quit()

    train = data[np.isin(data[:,0], train_names)]
    test = data[np.isin(data[:,0], test_names)]
    sort_data(train, sort_idx)
    sort_data(test, sort_idx)
    train, test = add_index_and_header(header, train, test, idx)
    # Write
    np.savetxt(script_dir + "/kaggle_dataset/" + basename + "_train.csv", train, delimiter=',', fmt='%s')
    np.savetxt(script_dir + "/kaggle_dataset/" + basename + "_test.csv", test, delimiter=',', fmt='%s')
    # Write masked version for the target
    if basename == 'data':
        np.savetxt(script_dir + "/kaggle_dataset/" + basename + "_test_masked.csv", test[:,:-1], delimiter=',', fmt='%s')
    # Hack to also merge all the structures to one file
    if basename == 'structures':
        data_with_header = add_index_and_header(header, data, data[:1], idx)
        np.savetxt(script_dir + "/kaggle_dataset/" + basename + ".csv", data_with_header, delimiter=',', fmt='%s')

def create_dataset():
    """
    Write csv files for the Kaggle dataset
    """
    # Get script location
    script_dir = os.path.abspath(os.path.dirname(__file__))

    data, contrib, pot, shield, mulliken, dipole, coords = parse_data(script_dir)

    # Get training and test molecules
    train_mols = np.loadtxt(script_dir + '/training_molecules.txt', dtype=str)
    test_mols = np.loadtxt(script_dir + '/testing_molecules.txt', dtype=str)

    # Split the data into train and test, change sorting, add index column if needed and write the csv files
    # Prevent some extra copies by cleaning up along the way and doing the big arrays last
    process_and_write(train_mols, test_mols, pot, script_dir, 'potential_energy', [0], idx=False)
    process_and_write(train_mols, test_mols, shield, script_dir, 'magnetic_shielding_tensors', [1,0], idx=False)
    process_and_write(train_mols, test_mols, mulliken, script_dir, 'mulliken_charges', [1,0], idx=False)
    process_and_write(train_mols, test_mols, dipole, script_dir, 'dipole_moments', [0], idx=False)
    process_and_write(train_mols, test_mols, coords, script_dir, 'structures', [0], idx=False)
    process_and_write(train_mols, test_mols, data, script_dir, 'data', [2,1,0], idx=True)
    process_and_write(train_mols, test_mols, contrib, script_dir, 'scalar_coupling_contributions', [2,1,0], idx=False)

if __name__ == "__main__":
    create_dataset()
