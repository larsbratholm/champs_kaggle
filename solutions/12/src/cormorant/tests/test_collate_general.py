import torch
import numpy as np

from cormorant.data.collate import batch_stack_general, batch_stack

num_batch = 10
max_atoms = 10

# Test list of ints:
intlist = torch.randint(1,max_atoms,(num_batch,)).tolist()

stacked = batch_stack_general(intlist)

print('int test:', stacked)
print()

# Test list of floats:
floatlist = torch.rand((num_batch,)).tolist()

stacked = batch_stack_general(floatlist)

print('float test:', stacked)
print()


# Test set of numpy arrays:
sizes = torch.randint(1,max_atoms,(num_batch,))

batch = [torch.rand(s,).numpy() for s in sizes]
stacked = batch_stack_general(batch)

print('Numpy test:')
print(stacked.dtype, max(sizes), stacked.shape)
print()


# Test scalars independent of atom number:
sizes = torch.randint(1,max_atoms,(num_batch,))

batch = [torch.rand(1)[0] for s in sizes]
stacked = batch_stack_general(batch)
stacked_old = batch_stack(batch)

assert (stacked == stacked_old).all()

print('scalar test:')

print(stacked)
print(stacked.shape)

print()

# Test scalars for each atom:
sizes = torch.randint(1,max_atoms,(num_batch,))

batch = [torch.rand(s,) for s in sizes]
stacked = batch_stack_general(batch)
stacked_old = batch_stack(batch)

assert (stacked == stacked_old).all()

print('scalar atom test:')

print([x.shape for x in batch])
# print(stacked)
print(stacked.shape)
print([(len(x.nonzero()) == s).item() for x, s in zip(stacked, sizes)])

print()

# Test vectors (extra dimensions) for each atom:
sizes = torch.randint(1,max_atoms,(num_batch,))

batch = [torch.rand(s,3) for s in sizes]
stacked = batch_stack_general(batch)
stacked_old = batch_stack(batch)

assert (stacked == stacked_old).all()

print('vector atom test:')

print([x.shape for x in batch])
# print(stacked)
print(max(sizes).item(), stacked.shape)
print([(len(x.nonzero())//3 == s).item() for x, s in zip(stacked, sizes)])

print()

# Test matrix (scalars) for each atom:
sizes = torch.randint(1,max_atoms,(num_batch,))

batch = [torch.rand(s,s) for s in sizes]
stacked = batch_stack_general(batch)
stacked_old = batch_stack(batch, edge_mat=True)

assert (stacked == stacked_old).all()

print('matrix atom scalar test:')

print([x.shape for x in batch])
# print(stacked)
print(max(sizes).item(), stacked.shape)
print([(len(x.nonzero()) == s*s).item() for x, s in zip(stacked, sizes)])

print()

# Test matrix (vectors) for each atom:
sizes = torch.randint(1,max_atoms,(num_batch,))

batch = [torch.rand(s,s,3) for s in sizes]
stacked = batch_stack_general(batch)
stacked_old = batch_stack(batch, edge_mat=True)

assert (stacked == stacked_old).all()

print('matrix atom vector test:')

print([x.shape for x in batch])
# print(stacked)
print(max(sizes).item(), stacked.shape)
print([(len(x.nonzero()) == s*s*3).item() for x, s in zip(stacked, sizes)])

print()


# Test failure mode if batch sizes disagree along two dimensions
sizes1, sizes2 = torch.randint(1,max_atoms,(num_batch,)), torch.randint(1,max_atoms,(num_batch,))

batch = [torch.rand(s1,s2) for s1, s2 in zip(sizes1, sizes2)]

print('matrix assert test:')
try:
    stacked = batch_stack_general(batch)
except AssertionError:
    print('assert passed!')



print()
