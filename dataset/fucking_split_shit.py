import numpy as np
import sys
import re

def parse(filename, mols):
    mols_set = set(mols)
    with open(filename) as f:
        print(f.readline(), end='')
        for line in f:
            if set(line.split(',')).intersection(mols_set):
                print(line, end='')


if __name__ == "__main__":
    filename = sys.argv[1]
    with open(sys.argv[2]) as f:
        mols = [line.strip() for line in f]

    parse(filename, mols)
