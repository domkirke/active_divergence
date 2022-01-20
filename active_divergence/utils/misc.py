from collections import OrderedDict
import omegaconf
import numpy as np, os, torch, sys, torch.distributions as dist, copy, bisect
sys.path.append('../')

def parse_slice(item, length):
    start = item.start or 0
    stop = item.stop or length
    start = start if start >= 0 else length + start
    stop = start if stop >= 0 else length + stop
    return start, stop, item.step


def checkdist(obj):
    if isinstance(obj, str):
        return getattr(dist, obj)
    elif issubclass(obj, dist.Distribution):
        return obj
    else:
        raise TypeError('obj %s does not seem to be a distribution')

def filter_nans(x: np.ndarray, y: np.ndarray = None):
    idxs = np.where(np.isnan(x))
    if y is None:
        x[idxs] = 0.
    else:
        x[idxs] = y[idxs]

def checklist(item, n=1, copy=False):
    """Repeat list elemnts
    """
    if not isinstance(item, (list, )):
        if copy:
            item = [copy.deepcopy(item) for _ in range(n)]
        elif isinstance(item, omegaconf.listconfig.ListConfig):
            item = list(item)
        else:
            item = [item]*n
    return item

def checktuple(item, n=1, copy=False):
    """Check tuple"""
    if not isinstance(item, tuple):
        if copy:
            item = tuple([copy.deepcopy(item) for _ in range(n)])
        elif isinstance(item, omegaconf.listconfig.ListConfig):
            item = tuple(item)
        else:
            item = tuple([item]*n)
    return item

def checkdir(directory):
    """Check directory existence. Create a directory if necessary
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)

def checktensor(tensor, dtype=None, allow_0d=True):
    if isinstance(tensor, list):
        return [checktensor(t, dtype=dtype) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple([checktensor(t, dtype=dtype) for t in tensor])
    elif isinstance(tensor, dict):
        return {k: checktensor(v, dtype=dtype) for k, v in tensor.items()}
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).to(dtype=dtype)
    elif torch.is_tensor(tensor):
        tensor = tensor.to(dtype=dtype)
        if tensor.ndim == 0 and not allow_0d:
            tensor = torch.Tensor([tensor])
        return tensor
    else:
        if hasattr(tensor, "__iter__"):
            tensor = torch.Tensor(tensor, dtype=dtype)
        else:
            tensor = torch.tensor(tensor, dtype=dtype)
        if tensor.ndim == 0 and not allow_0d:
            tensor = torch.Tensor([tensor])
        return tensor

def checknumpy(tensor):
    if isinstance(tensor, list):
        return [checknumpy(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple([checknumpy(t) for t in tensor])
    elif isinstance(tensor, dict):
        return {k: checknumpy(v) for k, v in tensor.items()}
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()

def print_stats(k, v):
    print(f"{k}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.3f}")

def print_module_grads(module):
    for k, v in module.named_parameters():
        v = v.grad
        if v is None:
            print(f'{k}: None')
        else:
            print_stats(k, v.grad)

def get_shape_from_ratio(n_item, target_ratio):
    i = np.ceil(np.sqrt(n_item / np.prod(target_ratio)))
    h, w = target_ratio[1] * i, target_ratio[0] * i
    return int(h), int(w)

def kronecker(A, B):
    """
    Kronecker product of two incoming matrices A and B
    Args:
        A (torch.Tensor): b x _ x _ matrix
        B (torch.Tensor): b x _ x _ matrix

    Returns:
        out (torch.Tensor): Kronkecker product
    """
    assert len(A.shape) == len(B.shape) == 3, "kroncker takes b x _ x _ matrices"
    requires_grad = A.requires_grad or B.requires_grad
    out = torch.zeros(A.shape[0], A.size(1)*B.size(1),  A.size(2)*B.size(2), requires_grad=requires_grad, device=A.device)
    for i in range(A.shape[0]):
        out[i] =  torch.einsum("ab,cd->acbd", A[i], B[i]).contiguous().view(A.size(1)*B.size(1),  A.size(2)*B.size(2))
    return out

def frame(tensor, wsize, hsize, dim, pad=False):
    if dim < 0:
        dim = tensor.ndim + dim
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if pad:
        pad(tensor)
    else:
        n_windows = (tensor.shape[dim] - wsize) // hsize
        if tensor.shape[dim] >= (n_windows + 1) * hsize + wsize:
            n_windows += 1
    shape = list(tensor.shape)
    strides = np.array(tensor.stride())
    new_stride = int(np.prod(strides[dim:][strides[dim:]>0]))
    shape = shape[:dim] + [n_windows, wsize] + shape[dim+1:]
    strides = list(strides[:dim]*hsize) + [hsize * new_stride] + list(strides[dim:])
    return torch.as_strided(tensor, tuple(shape), tuple(strides))

def overlap_add(array, wsize, hsize, window=None):
    batch_shape = array.shape[:-2]
    array_shape = (*batch_shape, hsize * array.shape[-2] + wsize)
    buffer = np.zeros(*batch_shape, array_shape)
    for i in range(array.shape[-2]):
        idx = [slice(None)] * len(batch_shape) + [i, slice(None)]
        current_slice = array.__getitem__(idx)
        buffer[..., i * hsize:i*hsize + wsize] += current_slice
    return buffer


def unwrap(tensor: torch.Tensor):
    """
    unwrap phase for tensors
    :param tensor: phase to unwrap (seq x spec_bin)
    :return: unwrapped phase
    """
    if isinstance(tensor, list):
        return [unwrap(t) for t in tensor]
    if tensor.ndimension() == 2:
        unwrapped = tensor.clone()
        diff = tensor[1:] - tensor[:-1]
        ddmod = (diff + torch.pi)%(2 * torch.pi) - torch.pi
        mask = (ddmod == -torch.pi).bitwise_and(diff > 0)
        ddmod[mask] = torch.pi
        ph_correct = ddmod - diff
        ph_correct[diff.abs() < torch.pi] = 0
        unwrapped[1:] = tensor[1:] + torch.cumsum(ph_correct, 1)
        return unwrapped
    else:
        return torch.stack([unwrap(tensor[i]) for i in range(tensor.size(0))], dim=0)


def fdiff(x, order=2):
    if order == 1:
        inst_f = torch.cat([x[0].unsqueeze(0), (x[1:] - x[:-1])/2], axis=0)
    elif order == 2:
        inst_f = torch.cat([x[0].unsqueeze(0), (x[2:] - x[:-2])/4, x[-1].unsqueeze(0)], axis=0)
    return inst_f


def fint(x, order=1):
    if order == 1:
        out = x
        out[1:] = out[1:] * 2
        if torch.is_tensor(x):
            out = torch.cumsum(out, axis=0)
        else:
            out = torch.cumsum(out, axis=0)
    elif order == 2:
        out = torch.zeros_like(x)
        out[0] = x[0]; out[-1] = x[-1]

        for i in range(2, x.shape[0], 2):
            out[i] = out[i-2] + 4 * x[i-1]
        for i in reversed(range(1, x.shape[0], 2)):
            out[i-2] = out[i] - 4 * x[i-1]
    return out


class ContinuousSlice(object):
    def __init__(self, *args):
        self.start = None
        self.stop = None
        self.step = None
        assert len(args) > 1
        if len(args) == 2:
            self.start, self.stop = args
        elif len(args) == 3:
            self.start, self.stop, self.step = args


class ContinuousList(object):
    def __init__(self, *args, append=False, drop_with_offset=None, default_values=None):
        if len(args) == 0:
            self._hash = {}
            self._ordered_values = []
        elif len(args) == 1:
            if isinstance(args[0], ContinuousList):
                self._hash = copy.copy(args[0]._hash)
                self._ordered_values = copy.copy(args[0]._ordered_values)
            elif isinstance(args[0], dict):
                # assert functools.reduce(lambda x, y: x and isinstance(y, float), args[0].keys(), True), \
                #     "dict keys must be floats"
                self._hash = args[0]
                self._ordered_values = sorted(list(self._hash.keys()))
            else:
                raise TypeError("ContinuousList must be initialised with ContinousList, dict, or a list of values.")
        else:
            self._hash = {i: args[i] for i in range(len(args))}
        self.append = append
        self.drop_with_offset = drop_with_offset

    def get_idx(self, key):
        idx = bisect.bisect_left(self._ordered_values, key)
        return idx

    def __iter__(self):
        _ord = self._ordered_values + [None]
        for i in range(len(self._ordered_values)):
            yield _ord[i], _ord[i+1], self._hash[_ord[i]]

    def __contains__(self, item):
        return item in self._ordered_values

    def __setitem__(self, key, value):
        # try:
        #     key = float(key)
        # except TypeError:
        #     raise TypeError('item assignement for ContinuousList is only valid for float')

        if self._hash.get(key) is None:
            idx = self.get_idx(key)
            self._ordered_values.insert(idx, key)
        else:
            if self.append:
                if isinstance(self._hash[key], list):
                    self._hash[key].append(value)
                    return
        self._hash[key] = value

    def __getitem__(self, item, drop_with_offset=None):
        drop_with_offset = drop_with_offset or self.drop_with_offset
        if isinstance(item, tuple):
            item = ContinuousSlice(*item)
        if torch.is_tensor(item):
            if len(item.shape) == 0:
                item = item.item()
        if hasattr(item, "__iter__"):
            if drop_with_offset == 'absolute':
                idxs = [self.get_idx(i) for i in item]
                return {i: self._hash[i] for i in idxs}
            elif drop_with_offset == 'relative':
                idxs = [self.get_idx(i) for i in item]
                return {i-min(idxs): self._hash[i] for i in idxs}
            else:
                return [self.__getitem__(i) for i in item]
        elif isinstance(item, (slice, ContinuousSlice)):
            start = item.start; end = item.stop
            if item.step is not None:
                raise NotImplementedError
            if start is None:
                hash_keys = self._ordered_values[:self.get_idx(end)]
            elif end is None:
                hash_keys = self._ordered_values[self.get_idx(start):]
            else:
                start_idx = self.get_idx(start); end_idx = self.get_idx(end)
                if start_idx == end_idx:
                    if start_idx < len(self._ordered_values):
                        hash_keys = [self._ordered_values[self.get_idx(start)]]
                    else:
                        hash_keys = [self._ordered_values[-1]]
                else:
                    hash_keys = self._ordered_values[self.get_idx(start):self.get_idx(end)]
            if drop_with_offset == "absolute":
                return {k: self._hash[k] for k in hash_keys}
            if drop_with_offset == "relative":
                return {k-hash_keys[0]: self._hash[k] for k in hash_keys}
            else:
                return [self._hash[k] for k in hash_keys]
        else:
            # item = float(item)
            idx = max(min(self.get_idx(item)-1, len(self._ordered_values)), 0)
            # print(idx)
            if drop_with_offset == "absolute":
                return {self._ordered_values[idx]: self._hash[self._ordered_values[idx]]}
            elif drop_with_offset == "relative":
                return {0: self._hash[self._ordered_values[idx]]}
            else:
                return self._hash[self._ordered_values[idx]]

    def __repr__(self):
        return str({i: self._hash[i] for i in self._ordered_values})

