
TYPES_LIST = [
    '1JHC',
    '1JHN',
    '2JHC',
    '2JHH',
    '2JHN',
    '3JHC',
    '3JHH',
    '3JHN',
]


TYPES_DICT = {
    '1JHC':0,
    '1JHN':1,
    '2JHC':2,
    '2JHH':3,
    '2JHN':4,
    '3JHC':5,
    '3JHH':6,
    '3JHN':7,
}

# (1/train.type.value_counts(normalize=True)).to_dict()
TYPES_WEIGHTS = {
    '3JHC': 3.0840914763777834,
    '2JHC': 4.083679473714663,
    '1JHC': 6.56617132965707,
    '3JHH': 7.886996686482305,
    '2JHH': 12.321966691002974,
    '3JHN': 27.991148634437998,
    '2JHN': 39.06104668226376,
    '1JHN': 107.42215713857435}
