"""
Create the Kaggle data from Gaussian output files
"""

import pybel
import numpy as np
import openbabel
import os
import tarfile

def mol_read_conn(mol):
    atoms = len(mol.atoms)
    conn_mat = np.zeros((atoms, atoms), dtype=np.int32)
    for x in range(atoms):
        a1 = x
        for nbr_atom in openbabel.OBAtomAtomIter(mol.atoms[x].OBAtom):
            a2 = nbr_atom.GetId()
            bond = mol.atoms[x].OBAtom.GetBond(nbr_atom)
            num = bond.GetBondOrder()
            conn_mat[a1][a2] = num
            conn_mat[a2][a1] = num

    return conn_mat

def conn_distance(conn_mat, max=100):

    # conn_mat = connectivity matrix, nxn (n = atoms) containing an integer>0 or a 0 to show if the two atoms are connected by a bond

    # define atoms from length of axis 0 of connectivity matrix
    atoms = conn_mat.shape[0]

    # define bond distance matrix where all bits are max distance away we're looking
    dist = max * np.ones((atoms, atoms), dtype=np.int32)

    # atoms_found is a list tracking things, first dim tracks the start point for connectivity, second dim tracks when an atom has been 'found' and the third dim tracks at what depth the atom was found
    # atoms_found actually tracks all the data needed to make the dist matrix, you just have to collapse its third dimension into the second but I kept it separate.

    for i in range(atoms):
        # It's written so you have a starting point atom (i) and look for atoms its connected to, then atoms connected to those ones and then atoms connected to the ones connected to the previous set of ones

        # Make a list of lists so the first index encodes for depth the atom was found at containing a list of atoms found at that depth
        atoms_found = [ [] for x in range(max)]
        # We make a normal list of all the atoms we haven't yet found to tell us what to loop over
        atoms_not_found = list(range(atoms)) # Need list of range as range is it's own object in python

        # Set the distance to itself as zero (this isn't set during the loops so it's best to do it here)
        dist[i][i] = 0
        # We add in atom i as the starting point (at depth 0)
        atoms_found[0].append(i)
        # So we have to remove atom i from the list of atoms to looks for connections with (as we've already "found" it)
        atoms_not_found.remove(i)

        # We keep track of how many bonds away we're at in our search using depth
        for depth in range(1,max):
            # If you didn't find anything on the last loop, it's time to leave
            if atoms_found[depth-1] == []:
                break
            # Look for connections between atoms we've found (depth - 1) bonds away from our start point, this is the loop through j
            for j in atoms_found[depth-1]:
                # To atoms we haven't found yet
                for k in atoms_not_found:
                    # We check if these pairs of atoms are connected or not
                    if not conn_mat[j][k] == 0:
                        # If they are we know that atoms connected to atoms (depth -1) bonds away from our start point are (depth) bonds away from our start point
                        dist[i][k] = depth
                        atoms_found[depth].append(k)

            # At the end of looking at this number of bonds away (depth) we remove the things we found at that depth
            # We have to separate this from the other loop because of list slippage
            for atom in atoms_found[depth]:
                try:
                    atoms_not_found.remove(atom)
                except:
                    continue
    return dist

def mol_pathway_finder(mol, atype1, atype2, coupling_length):
    atoms = len(mol.atoms)
    pathways = []
    for x in range(atoms):
        a1 = x
        if atype1 != mol.atoms[x].atomicnum:
            continue
        for nbr_atom in openbabel.OBAtomAtomIter(mol.atoms[x].OBAtom):
            a2 = nbr_atom.GetId()

            if coupling_length == 1 and atype2 == nbr_atom.GetAtomicNum():
                pathways.append([a1,a2])
                continue

            for nbr2_atom in openbabel.OBAtomAtomIter(nbr_atom):
                a3 = nbr2_atom.GetId()
                if coupling_length == 2 and atype2 == nbr2_atom.GetAtomicNum():
                    pathways.append([a1, a2, a3])
                    continue

                for nbr3_atom in openbabel.OBAtomAtomIter(nbr2_atom):
                    a4 = nbr3_atom.GetId()
                    if coupling_length == 3  and atype2 == nbr3_atom.GetAtomicNum():
                        pathways.append([a1, a2, a3, a4])
                        continue

                    for nbr4_atom in openbabel.OBAtomAtomIter(nbr3_atom):
                        a5 = nbr4_atom.GetId()
                        if coupling_length == 4  and atype2 == nbr4_atom.GetAtomicNum():
                            pathways.append([a1, a2, a3, a4, a5])
                            continue

                        for nbr5_atom in openbabel.OBAtomAtomIter(nbr4_atom):
                            a6 = nbr5_atom.GetId()
                            if coupling_length == 5 and atype2 == nbr5_atom.GetAtomicNum():
                                pathways.append([a1, a2, a3, a4, a5, a6])
                                continue

    path_array = np.zeros((len(pathways), coupling_length+1), dtype=int)
    for i in range(len(pathways)):
        for j in range(coupling_length + 1):
            path_array[i][j] = pathways[i][j]

    return path_array

def mol_read_type(mol):
    type_list = []
    type_array = np.zeros(len(mol.atoms), dtype=np.int32)
    Periodic_table = Get_periodic_table()
    for i in range(len(mol.atoms)):
        type = int(mol.atoms[i].atomicnum)
        type_array[i] = type
        type_list.append(Periodic_table[type])

    return type_list, type_array

def Get_periodic_table():
    periodic_table=[
        '',
        'H',
        'He',
        'Li',
        'Be',
        'B',
        'C',
        'N',
        'O',
        'F',
        'Ne',
        'Na',
        'Mg',
        'Al',
        'Si',
        'P',
        'S',
        'Cl',
        'Ar',
        'K',
        'Ca',
        'Sc',
        'Ti',
        'V',
        'Cr',
        'Mn',
        'Fe',
        'Co',
        'Ni',
        'Cu',
        'Zn',
        'Ga',
        'Ge',
        'As',
        'Se',
        'Br',
        'Kr',
        'Rb',
        'Sr',
        'Y',
        'Zr',
        'Nb',
        'Mo',
        'Tc',
        'Ru',
        'Rh',
        'Pd',
        'Ag',
        'Cd',
        'In',
        'Sn',
        'Sb',
        'Te',
        'I',
        'Xe',
        'Cs',
        'Ba',
        'La',
        'Ce',
        'Pr',
        'Nd',
        'Pm',
        'Sm',
        'Eu',
        'Gd',
        'Tb',
        'Dy',
        'Ho',
        'Er',
        'Tm',
        'Yb',
        'Lu',
        'Hf',
        'Ta',
        'W',
        'Re',
        'Os',
        'Ir',
        'Pt',
        'Au',
        'Hg',
        'Tl',
        'Pb',
        'Bi',
        'Po',
        'At',
        'Rn',
        'Fr',
        'Ra',
        'Ac',
        'Th',
        'Pa',
        'U',
        'Np',
        'Pu',
        'Am',
        'Cm',
        'Bk',
        'Cf',
        'Es',
        'Fm',
        'Md',
        'No',
        'Lr',
        'Rf',
        'Db',
        'Sg',
        'Bh',
        'Hs',
        'Mt',
        'Ds',
        'Rg',
        'Cn',
        'Nh',
        'Fl',
        'Mc',
        'Lv',
        'Ts',
        'Og',
        ]
    return periodic_table

def jread(filename, atomnumber):

    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    with open(filename) as f:
        skip = True
        for line in f:
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

def mol_read_xyz(mol):
    xyz_array = np.zeros((len(mol.atoms),3), dtype=np.float64)
    for i in range(len(mol.atoms)):
        xyz_array[i][0] = float(mol.atoms[i].coords[0])
        xyz_array[i][1] = float(mol.atoms[i].coords[1])
        xyz_array[i][2] = float(mol.atoms[i].coords[2])

    # Return array is zero indexed
    return xyz_array

def is_empty(generator):
    for item in generator:
        return False
    return True

def read_rotational_constants(filename):
    rotational_constants = None
    with open(filename) as f:
        for line in f:
            if line.startswith(" Rotational constants"):
                rotational_constants = np.asarray(line.split()[3:], dtype=float)
                break
    if rotational_constants is None or rotational_constants.size != 3:
        raise SystemExit("Error reading rotational constants in %s" % filename)
    return rotational_constants

def read_dipole_moment(filename):
    dipole_moment = None
    skip_line = False
    with open(filename) as f:
        for line in f:
            if line.startswith(" Dipole moment"):
                skip_line = True
            elif skip_line:
                dipole_moment = np.asarray(line.split()[1:6:2], dtype=float)
                break
    if dipole_moment is None or dipole_moment.size != 3:
        raise SystemExit("Error reading dipole moment in %s" % filename)
    return dipole_moment

def read_potential_energy(filename):
    potential_energy = None
    with open(filename) as f:
        for line in f:
            if line.startswith(" SCF Done:"):
                potential_energy = float(line.split()[4])
                break
    if potential_energy is None:
        raise SystemExit("Error reading potential energy in %s" % filename)
    return potential_energy

def read_magnetic_shielding_tensor(filename):
    shieldings = []
    atom_shieldings = []
    skip = True
    with open(filename) as f:
        for line in f:
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
        raise SystemExit("Error reading magnetic shielding tensor in %s" % filename)
    return shieldings_array

def read_mulliken_charges(filename):
    mulliken_charges = []
    c = -1
    with open(filename) as f:
        for line in f:
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
        raise SystemExit("Error reading mulliken charges in %s" % filename)
    return np.asarray(mulliken_charges, dtype=float)

def jread_fc(filename, atomnumber):
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    with open(filename) as f:
        skip = True
        for line in f:
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

def jread_sd(filename, atomnumber):
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    with open(filename) as f:
        skip = True
        for line in f:
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

def jread_pso(filename, atomnumber):
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    with open(filename) as f:
        skip = True
        for line in f:
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

def jread_dso(filename, atomnumber):
    couplings = np.zeros((atomnumber, atomnumber), dtype=float)
    with open(filename) as f:
        skip = True
        for line in f:
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

def create_dataset():#output_folder, xyz_folder):
    #output_files = {}
    #for name in ("data", "potential_energy", "magnetic_shielding_tensors",
    #        "scalar_coupling_contributions", "mulliken_charges", "dipole_moments"):
    #    output_files[name] = open(output_folder + "/%s.txt" % name, "w")

    ##headers
    #output_files["data"].write("molecule_name,atom_index_0,atom_index_1,type,scalar_coupling_constant\n")
    #output_files["potential_energy"].write("molecule_name,potential_energy\n")
    #output_files["magnetic_shielding_tensors"].write("molecule_name,atom_index,XX,YX,ZX,XY,YY,ZY,XZ,YZ,ZZ\n")
    #output_files["scalar_coupling_contributions"].write("molecule_name,atom_index_0,atom_index_1,type,fc,sd,pso,dso\n")
    #output_files["mulliken_charges"].write("molecule_name,atom_index,mulliken_charge\n")
    #output_files["dipole_moments"].write("molecule_name,X,Y,Z\n")

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

    for member in log_tar.getmembers():
        if member.isfile():
            basename = member.name.split("/")[-1].split(".")[0]
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
            #conn_mat = mol_read_conn(mol)
            #mol = next(pybel.readstring('xyz', xyz_string))
            #print(len(mol.atoms))
            #conn_mat = mol_read_conn(mol)
            #conn_dist = conn_distance(conn_mat, 4)
            #print(conn_mat.shape[0])
            #conn_mat = mol_read_conn(mol)
            #conn_dist = conn_distance(conn_mat, 4)
            #atom_types = np.asarray([x.atomicnum for x in mol.atoms])

            # Read all the scalar coupling constants
            # and their decomposition
            j_array = jread(filename, n_atoms)
            j_array_fc = jread_fc(filename, n_atoms)
            j_array_sd = jread_sd(filename, n_atoms)
            j_array_pso = jread_pso(filename, n_atoms)
            j_array_dso = jread_dso(filename, n_atoms)

            #c = 0
            #atom1_loc = np.where(atom_types == 1)[0]
            #for name in output_files.keys():
            #    if name[1] != "J":
            #        continue
            #    coupling_length = int(name[0])
            #    if name[-1] == "H":
            #        atom_index = 1
            #    elif name[-1] == "C":
            #        atom_index = 6
            #    elif name[-1] == "N":
            #        atom_index = 7
            #    else:
            #        raise SystemExit("some error")

            #    atom2_loc = np.where(atom_types == atom_index)[0]

            #    for i in atom1_loc:
            #        for j in atom2_loc:
            #            if atom_index == 1 and i >= j:
            #                continue

            #            if conn_dist[i,j] == coupling_length:
            #                line = "%s,%d,%d,%f\n" % (basename, i, j, j_array[i,j])
            #                extended_line = "%s,%d,%d,%s,%.6f\n" % (basename, i, j, name, j_array[i,j])
            #                output_files[name].write(line)
            #                output_files["data"].write(extended_line)
            #                c += 1
            #                output_files["scalar_coupling_contributions"].write(
            #                        "%s,%d,%d,%s,%.6f,%.6f,%.6f,%.6f\n" % (basename, i, j, name, 
            #                            j_array_fc[i,j], j_array_sd[i,j], j_array_pso[i,j], j_array_dso[i,j]))

            #if c == 0:
            #    print("No couplings in:", filename)
            #    continue

            ##rotational_constants = read_rotational_constants(filename)
            ##line = "%s,%.7f,%.7f,%.7f\n" % (basename, *rotational_constants)
            ##output_files["rotational_constants"].write(line)

            #potential_energy = read_potential_energy(filename)
            #line = "%s %.7f\n" % (basename, potential_energy)
            #output_files["potential_energy"].write(line)

            #shielding_tensor = read_magnetic_shielding_tensor(filename)
            #for j in range(shielding_tensor.shape[0]):
            #    line = ("%s,%d" + ",%.4f" * 9 + "\n") % (basename, j, *shielding_tensor[j])
            #    output_files["magnetic_shielding_tensors"].write(line)

            #mulliken_charges = read_mulliken_charges(filename)
            #for j in range(len(mulliken_charges)):
            #    line = ("%s,%d,%.6f\n") % (basename, j, mulliken_charges[j])
            #    output_files["mulliken_charges"].write(line)

            #dipole_moment = read_dipole_moment(filename)
            #line = "%s,%.4f,%.4f,%.4f\n" % (basename, *dipole_moment)
            #output_files["dipole_moments"].write(line)


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
    create_dataset()#filenames, output_folder, xyz_folder)


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
