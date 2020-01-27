import openbabel as bb
from common import *

conversion = bb.OBConversion()
conversion.SetInFormat('xyz')

def read_molecule(molecule_name):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f'{script_dir}/../../data/xyz/{molecule_name}.xyz') as f:
        molecule = bb.OBMol()
        conversion.ReadString(molecule, f.read())

    return molecule

GLOBAL_FEATURES_COLUMNS = [
    "NumAtoms",
    "NumBonds",
    "NumHvyAtoms",
    "NumResidues",
    "NumRotors",
    "GetMolWt",
    "GetEnergy",
    "GetExactMass",
    "GetTotalCharge",
    "GetTotalSpinMultiplicity",
    "IsChiral",
    "NumConformers"
]
def get_global_features(mol):
    global_features = [
        mol.NumAtoms(),
        mol.NumBonds(),
        mol.NumHvyAtoms(),
        mol.NumResidues(),
        mol.NumRotors(),
        mol.GetMolWt(),
        mol.GetEnergy(),
        mol.GetExactMass(),
        mol.GetTotalCharge(),
        mol.GetTotalSpinMultiplicity(),
        mol.IsChiral(),
        mol.NumConformers()
    ]
    return global_features


ATOM_FEATURES_COLUMNS = [
    "GetFormalCharge",
    "GetSpinMultiplicity",
    "GetAtomicMass",
    "GetExactMass",
    "GetAtomicNum",
    "GetValence",
    "GetHyb",
    "GetImplicitValence",
    "GetHvyValence",
    "GetHeteroValence",
    "GetPartialCharge",
    "CountFreeOxygens",
    "ImplicitHydrogenCount",
    "ExplicitHydrogenCount",
    "MemberOfRingCount",
    "MemberOfRingSize",
    "CountRingBonds",
    "SmallestBondAngle",
    "AverageBondAngle",
    "BOSum",
    "HasResidue",
    "IsAromatic",
    "IsInRing",
    "IsHeteroatom",
    "IsNotCorH",
    "IsCarboxylOxygen",
    "IsPhosphateOxygen",
    "IsSulfateOxygen",
    "IsNitroOxygen",
    "IsAmideNitrogen",
    "IsPolarHydrogen",
    "IsNonPolarHydrogen",
    "IsAromaticNOxide",
    "IsChiral",
    "IsAxial",
    "IsHbondAcceptor",
    "IsHbondDonor",
    "IsHbondDonorH",
    "HasAlphaBetaUnsat",
    "HasNonSingleBond",
    "HasSingleBond",
    "HasDoubleBond",
    "HasAromaticBond"
]
def get_atom_features(atom):
    return [
        atom.GetFormalCharge(),
        atom.GetSpinMultiplicity(),
        atom.GetAtomicMass(),
        atom.GetExactMass(),
        atom.GetAtomicNum(),
        atom.GetValence(),
        atom.GetHyb(),
        atom.GetImplicitValence(),
        atom.GetHvyValence(),
        atom.GetHeteroValence(),
        atom.GetPartialCharge(),
        atom.CountFreeOxygens(),
        atom.ImplicitHydrogenCount(),
        atom.ExplicitHydrogenCount(),
        atom.MemberOfRingCount(),
        atom.MemberOfRingSize(),
        atom.CountRingBonds(),
        atom.SmallestBondAngle(),
        atom.AverageBondAngle(),
        atom.BOSum(),
        atom.HasResidue(),
        atom.IsAromatic(),
        atom.IsInRing(),
        atom.IsHeteroatom(),
        atom.IsNotCorH(),
        atom.IsCarboxylOxygen(),
        atom.IsPhosphateOxygen(),
        atom.IsSulfateOxygen(),
        atom.IsNitroOxygen(),
        atom.IsAmideNitrogen(),
        atom.IsPolarHydrogen(),
        atom.IsNonPolarHydrogen(),
        atom.IsAromaticNOxide(),
        atom.IsChiral(),
        atom.IsAxial(),
        atom.IsHbondAcceptor(),
        atom.IsHbondDonor(),
        atom.IsHbondDonorH(),
        atom.HasAlphaBetaUnsat(),
        atom.HasNonSingleBond(),
        atom.HasSingleBond(),
        atom.HasDoubleBond(),
        atom.HasAromaticBond()
    ]

BOND_FEATURES_COLUMNS = [
    "IsBond",
    "GetBondOrder",
    "GetEquibLength",
    "GetLength",
    "IsAromatic",
    "IsInRing",
    "IsRotor",
    "IsAmide",
    "IsPrimaryAmide",
    "IsSecondaryAmide",
    "IsTertiaryAmide",
    "IsEster",
    "IsCarbonyl",
    "IsSingle",
    "IsDouble",
    "IsTriple",
    "IsClosure",
    "IsUp",
    "IsDown",
    "IsCisOrTrans",
    "IsDoubleBondGeometry"
]
def get_bond_features(bond):
    if bond is None:
        return [0] * len(BOND_FEATURES_COLUMNS)
    else:
        return [
            1,
            bond.GetBondOrder(),
            bond.GetEquibLength(),
            bond.GetLength(),
            bond.IsAromatic(),
            bond.IsInRing(),
            bond.IsRotor(),
            bond.IsAmide(),
            bond.IsPrimaryAmide(),
            bond.IsSecondaryAmide(),
            bond.IsTertiaryAmide(),
            bond.IsEster(),
            bond.IsCarbonyl(),
            bond.IsSingle(),
            bond.IsDouble(),
            bond.IsTriple(),
            bond.IsClosure(),
            bond.IsUp(),
            bond.IsDown(),
            bond.IsCisOrTrans(),
            bond.IsDoubleBondGeometry()
        ]
    


def compute_features(molecule):
    mol = read_molecule(molecule)
    
    global_features = get_global_features(mol)
    atom_features = []
    bond_features = []
    
    for atom_index_i in range(mol.NumAtoms()):
        atom_i = mol.GetAtomById(atom_index_i)
        atom_features.append(get_atom_features(atom_i))
        
        for atom_index_j in range(mol.NumAtoms()):
            atom_j = mol.GetAtomById(atom_index_j)
            
            if atom_index_i < atom_index_j:
                bond = mol.GetBond(atom_i, atom_j)
                bond_features.append(get_bond_features(bond))
    
    return np.array(global_features).reshape(1, -1), np.array(atom_features), np.array(bond_features)