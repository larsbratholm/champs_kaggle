"""
Scripts to convert the extended xyz (exyz) format of QM9 into
valid xyz as well as input files (com) for gaussian NMR computations
"""

import numpy as np
import tarfile
import os

def fix_broken_format(x):
    """
    Fix the broken 
    """
    if "*^" in x:
        tokens = x.split("*^")
        a = float(tokens[0])
        b = int(tokens[1])
        return a * 10**b
    else:
        return float(x)

def exyz_to_com(lines, output_filename):
    """
    Convert exyz to com files
    """

    with open(output_filename, "w") as f:
        n = int(lines[0])

        title = output_filename.split("/")[-1].split(".")[0]
        f.write("%mem=4gb\n%nproc=1\n#T nmr(spinspin,readatoms) NoSymmetry b3lyp/6-31g(2df,p)\n\n")
        f.write("%s\n\n0 1\n" % title)

        for line in lines[2:2+n]:
            tokens = line.split()
            atype = tokens[0]
            x = tokens[1]
            y = tokens[2]
            z = tokens[3]

            x = fix_broken_format(x)
            y = fix_broken_format(y)
            z = fix_broken_format(z)

            f.write("%s %.10f %.10f %.10f\n" % (atype, x, y, z))
        f.write("\natoms=H,C,N,F\n\n\n")

def exyz_to_xyz(lines, output_filename):
    """
    Convert exyz to xyz files
    """

    with open(output_filename, "w") as f:
        n = int(lines[0])

        f.write("%d\n" % n)

        for line in lines[2:2+n]:
            f.write("\n")
            tokens = line.split()
            atype = tokens[0]
            x = tokens[1]
            y = tokens[2]
            z = tokens[3]

            x = fix_broken_format(x)
            y = fix_broken_format(y)
            z = fix_broken_format(z)

            f.write("%s %.10f %.10f %.10f" % (atype, x, y, z))

if __name__ == "__main__":
    # Get script location
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Read the tar file containing all the qm9 exyz files
    try:
        tar = tarfile.open(script_dir + "/data/qm9_exyz.tar.gz", "r:gz")
    except FileNotFoundError:
        print("qm9_exyz.tar.gz not found.")
        quit()

    for member in tar.getmembers():
        if member.isfile():
            basename = member.name.split("/")[-1].split(".")[0]
            with tar.extractfile(member) as f:
                # tarfile doesn't decode the object, so just do it manually
                exyz_lines = [byte.decode('utf-8') for byte in f.readlines()]
            exyz_to_com(exyz_lines, script_dir + "/gaussian_input_files/" + basename + ".com")
            exyz_to_xyz(exyz_lines, script_dir + "/xyz/" + basename + ".xyz")
