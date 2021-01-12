import numpy as np
import torch
from functools import reduce  # Required in Python 3
import operator


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def multi_index_to_single(tensor, index):
    i = 0
    return torch.stack([index[i] * prod([tensor.shape[j] for j in range(i + 1, tensor.ndim)]) + index[i + 1] for i in range(len(index) - 1)]).sum(0)
    # return torch.stack([torch.Tensor([index[i] * prod([tensor.shape[j] for j in range(i + 1, tensor.ndim)]) + index[i + 1]]) for i in range(len(index) - 1)]).sum(0)


def add_at(tensor_a, index, tensor_b):
    index_flat = multi_index_to_single(tensor_a, index)
    # print(tensor_b.flatten().shape)
    # print(index_flat.max())
    # print(torch.index_select(tensor_a.flatten(), 0, index_flat))
    # torch.index_add(tensor_a.flatten().float(), 0, index_flat, tensor_b.flatten())
    # print(tensor_a.device, tensor_b.device, index_flat.device)
    return tensor_a.flatten().index_add_(0, index_flat, tensor_b.flatten()).reshape(tensor_a.shape)



# gvi = np.random.randn(5, 5) + np.random.randn(5, 5) * 1j
gvi = np.random.randn(5, 5) + np.random.randn(5, 5) * 1j
# tensor(4067017.2500) tensor(141988016.) torch.float32 torch.Size([5982336])
# tensor(48913.6250) tensor(22521834.) torch.float32 torch.Size([5982336])

gvi = np.zeros([2300, 2300], dtype=complex)
visg = 141988016 * np.random.randn(5982336) + 4067017 + 22521834j * np.random.randn(5982336) + 48913j

undxi = np.random.randint(0, 2300, size=(len(visg)))
vndxi = np.random.randint(0, 2300, size=(len(visg)))

gvi_t = torch.from_numpy(gvi)

np.add.at(gvi, (undxi, vndxi), visg)

visg_t = torch.from_numpy(visg)
undxi = torch.from_numpy(undxi)
vndxi = torch.from_numpy(vndxi)

gvi_tr = gvi_t.real
gvi_ti = gvi_t.imag

visg_tr = visg_t.real
visg_ti = visg_t.imag

add_at(gvi_tr, (undxi, vndxi), visg_tr)
add_at(gvi_ti, (undxi, vndxi), visg_ti)

gvi_t = torch.view_as_complex(torch.stack([gvi_tr, gvi_ti], dim=-1))


assert np.isclose(gvi_t, gvi).all()


