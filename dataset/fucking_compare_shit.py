import numpy as np
import sys
import copy

def parse_shit(filename):
    d = {}
    lines = 0
    with open(filename) as f:
        header = f.readline()
        c = 0
        if 'id' in header:
            c = 1
        for line in f:
            lines += 1
            tokens = line.split(',')
            name = tokens[0+c]
            idx0 = int(tokens[1+c])
            idx1 = int(tokens[2+c])
            value = float(tokens[4+c])

            if name not in d: d[name] = {}
            pair = (idx0, idx1)
            if pair in d[name]:
                print("duplicate", name, pair)
            d[name][pair] = value
    return d

def compare_shit(filenames):
    d1 = parse_shit(filenames[0])
    d2 = parse_shit(filenames[1])

    d1_remaining = copy.deepcopy(d1)
    d2_remaining = copy.deepcopy(d2)

    for name, subdict in d1.items():
        if name in d2:
            for pair, value in subdict.items():
                if pair in d2[name]:
                    if abs(value - d2[name][pair]) < 1e-2:
                        pass
                    else:
                        print("wrong", name, pair, value, d2[name][pair])
                    d1_remaining[name].pop(pair)
                    d2_remaining[name].pop(pair)
                else:
                    print("missing pair", name, pair)
        else:
            print("missing name", name)
            d1_remaining.pop(name)
            continue
        if len(d2_remaining[name]) != 0:
            print("extra", name, d2_remaining[name])
        d1_remaining.pop(name)
        d2_remaining.pop(name)
    for name in d2_remaining:
        for pair in d2_remaining[name]:
            print("extra", name, pair, d2_remaining[name][pair])


if __name__ == "__main__":
    filenames = sys.argv[1:]
    compare_shit(filenames)
