import pprint
import warnings
from collections.abc import Collection, Iterable, Iterator, KeysView, Sequence
from copy import deepcopy
from numbers import Number
from types import EllipsisType
from typing import (
    Any,
    Protocol,
    Self,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import torch
from deepdiff import DeepDiff

_SingleIndexType = slice | int | EllipsisType
IndexType = np.ndarray | _SingleIndexType | list[_SingleIndexType] | tuple[_SingleIndexType, ...]
TBatch = TypeVar("TBatch", bound="BatchProtocol")
arr_type = torch.Tensor | np.ndarray


def _is_batch_set(obj: Any) -> bool:
    # Batch set is a list/tuple of dict/Batch objects,
    # or 1-D np.ndarray with object type,
    # where each element is a dict/Batch object
    if isinstance(obj, np.ndarray):  # most often case
        # "for element in obj" will just unpack the first dimension,
        # but obj.tolist() will flatten ndarray of objects
        # so do not use obj.tolist()
        if obj.shape == ():
            return False
        return obj.dtype == object and all(isinstance(element, dict | Batch) for element in obj)
    if (
        isinstance(obj, list | tuple)
        and len(obj) > 0
        and all(isinstance(element, dict | Batch) for element in obj)
    ):
        return True
    return False


def _is_scalar(value: Any) -> bool:
    # check if the value is a scalar
    # 1. python bool object, number object: isinstance(value, Number)
    # 2. numpy scalar: isinstance(value, np.generic)
    # 3. python object rather than dict / Batch / tensor
    # the check of dict / Batch is omitted because this only checks a value.
    # a dict / Batch will eventually check their values
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and not value.shape
    # np.asanyarray will cause dead loop in some cases
    return np.isscalar(value)


def _is_number(value: Any) -> bool:
    # isinstance(value, Number) checks 1, 1.0, np.int(1), np.float(1.0), etc.
    # isinstance(value, np.nummber) checks np.int32(1), np.float64(1.0), etc.
    # isinstance(value, np.bool_) checks np.bool_(True), etc.
    # similar to np.isscalar but np.isscalar('st') returns True
    return isinstance(value, Number | np.number | np.bool_)


def _to_array_with_correct_type(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray) and issubclass(obj.dtype.type, np.bool_ | np.number):
        return obj  # most often case
    # convert the value to np.ndarray
    # convert to object obj type if neither bool nor number
    # raises an exception if array's elements are tensors themselves
    try:
        obj_array = np.asanyarray(obj)
    except ValueError:
        obj_array = np.asanyarray(obj, dtype=object)
    if not issubclass(obj_array.dtype.type, np.bool_ | np.number):
        obj_array = obj_array.astype(object)
    if obj_array.dtype == object:
        # scalar ndarray with object obj type is very annoying
        # a=np.array([np.array({}, dtype=object), np.array({}, dtype=object)])
        # a is not array([{}, {}], dtype=object), and a[0]={} results in
        # something very strange:
        # array([{}, array({}, dtype=object)], dtype=object)
        if not obj_array.shape:
            obj_array = obj_array.item(0)
        elif all(isinstance(arr, np.ndarray) for arr in obj_array.reshape(-1)):
            return obj_array  # various length, np.array([[1], [2, 3], [4, 5, 6]])
        elif any(isinstance(arr, torch.Tensor) for arr in obj_array.reshape(-1)):
            raise ValueError("Numpy arrays of tensors are not supported yet.")
    return obj_array


def create_value(
    inst: Any,
    size: int,
    stack: bool = True,
) -> Union["Batch", np.ndarray, torch.Tensor]:
    """Create empty place-holders according to inst's shape.

    :param stack: whether to stack or to concatenate. E.g. if inst has shape of
        (3, 5), size = 10, stack=True returns an np.array with shape of (10, 3, 5),
        otherwise (10, 5)
    """
    has_shape = isinstance(inst, np.ndarray | torch.Tensor)
    is_scalar = _is_scalar(inst)
    if not stack and is_scalar:
        # should never hit since it has already checked in Batch.cat_ , here we do not
        # consider scalar types, following the behavior of numpy which does not support
        # concatenation of zero-dimensional arrays (scalars)
        raise TypeError(f"cannot concatenate with {inst} which is scalar")
    if has_shape:
        shape = (size, *inst.shape) if stack else (size, *inst.shape[1:])
    if isinstance(inst, np.ndarray):
        target_type = (
            inst.dtype.type if issubclass(inst.dtype.type, np.bool_ | np.number) else object
        )
        return np.full(shape, fill_value=None if target_type == object else 0, dtype=target_type)
    if isinstance(inst, torch.Tensor):
        return torch.full(shape, fill_value=0, device=inst.device, dtype=inst.dtype)
    if isinstance(inst, dict | Batch):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = create_value(val, size, stack=stack)
        return zero_batch
    if is_scalar:
        return create_value(np.asarray(inst), size, stack=stack)
    # fall back to object
    return np.array([None for _ in range(size)], object)


def _assert_type_keys(keys: Iterable[str]) -> None:
    assert all(isinstance(key, str) for key in keys), f"keys should all be string, but got {keys}"


def _parse_value(obj: Any) -> Union["Batch", np.ndarray, torch.Tensor] | None:
    if isinstance(obj, Batch):  # most often case
        return obj
    if (
        (isinstance(obj, np.ndarray) and issubclass(obj.dtype.type, np.bool_ | np.number))
        or isinstance(obj, torch.Tensor)
        or obj is None
    ):  # third often case
        return obj
    if _is_number(obj):  # second often case, but it is more time-consuming
        return np.asanyarray(obj)
    if isinstance(obj, dict):
        return Batch(obj)
    if (
        not isinstance(obj, np.ndarray)
        and isinstance(obj, Collection)
        and len(obj) > 0
        and all(isinstance(element, torch.Tensor) for element in obj)
    ):
        try:
            obj = cast(list[torch.Tensor], obj)
            return torch.stack(obj)
        except RuntimeError as exception:
            raise TypeError(
                "Batch does not support non-stackable iterable"
                " of torch.Tensor as unique value yet.",
            ) from exception
    if _is_batch_set(obj):
        obj = Batch(obj)  # list of dict / Batch
    else:
        # None, scalar, normal obj list (main case)
        # or an actual list of objects
        try:
            obj = _to_array_with_correct_type(obj)
        except ValueError as exception:
            raise TypeError(
                "Batch does not support heterogeneous list/tuple of tensors as unique value yet.",
            ) from exception
    return obj


def alloc_by_keys_diff(
    meta: "BatchProtocol",
    batch: "BatchProtocol",
    size: int,
    stack: bool = True,
) -> None:
    """Creates place-holders inside meta for keys that are in batch but not in meta.

    This mainly is an internal method, use it only if you know what you are doing.
    """
    for key in batch.get_keys():
        if key in meta.get_keys():
            if isinstance(meta[key], Batch) and isinstance(batch[key], Batch):
                alloc_by_keys_diff(meta[key], batch[key], size, stack)
            elif isinstance(meta[key], Batch) and len(meta[key].get_keys()) == 0:
                meta[key] = create_value(batch[key], size, stack)
        else:
            meta[key] = create_value(batch[key], size, stack)


# Note: This is implemented as a protocol because the interface
# of Batch is always extended by adding new fields. Having a hierarchy of
# protocols building off this one allows for type safety and IDE support despite
# the dynamic nature of Batch
@runtime_checkable
class BatchProtocol(Protocol):
    """The internal data structure in Tianshou.

    Batch is a kind of supercharged array (of temporal data) stored individually in a
    (recursive) dictionary of objects that can be either numpy arrays, torch tensors, or
    batches themselves. It is designed to make it extremely easily to access, manipulate
    and set partial view of the heterogeneous data conveniently.

    For a detailed description, please refer to :ref:`batch_concept`.
    """

    @property
    def shape(self) -> list[int]:
        ...

    def __setattr__(self, key: str, value: Any) -> None:
        ...

    def __getattr__(self, key: str) -> Any:
        ...

    def __contains__(self, key: str) -> bool:
        ...

    def __getstate__(self) -> dict:
        ...

    def __setstate__(self, state: dict) -> None:
        ...

    @overload
    def __getitem__(self, index: str) -> Any:
        ...

    @overload
    def __getitem__(self, index: IndexType) -> Self:
        ...

    def __getitem__(self, index: str | IndexType) -> Any:
        ...

    def __setitem__(self, index: str | IndexType, value: Any) -> None:
        ...

    def __iadd__(self, other: Self | Number | np.number) -> Self:
        ...

    def __add__(self, other: Self | Number | np.number) -> Self:
        ...

    def __imul__(self, value: Number | np.number) -> Self:
        ...

    def __mul__(self, value: Number | np.number) -> Self:
        ...

    def __itruediv__(self, value: Number | np.number) -> Self:
        ...

    def __truediv__(self, value: Number | np.number) -> Self:
        ...

    def __repr__(self) -> str:
        ...

    def __iter__(self) -> Iterator[Self]:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    @staticmethod
    def to_numpy(batch: TBatch) -> TBatch:
        """Change all torch.Tensor to numpy.ndarray and return a new Batch."""
        ...

    def to_numpy_(self) -> None:
        """Change all torch.Tensor to numpy.ndarray in-place."""
        ...

    @staticmethod
    def to_torch(
        batch: TBatch,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> TBatch:
        """Change all numpy.ndarray to torch.Tensor and return a new Batch."""
        ...

    def to_torch_(
        self,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        """Change all numpy.ndarray to torch.Tensor in-place."""
        ...

    def cat_(self, batches: Self | Sequence[dict | Self]) -> None:
        """Concatenate a list of (or one) Batch objects into current batch."""
        ...

    @staticmethod
    def cat(batches: Sequence[dict | TBatch]) -> TBatch:
        """Concatenate a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros with appropriate shapes. E.g.
        ::

            >>> a = Batch(a=np.zeros([3, 4]), common=Batch(c=np.zeros([3, 5])))
            >>> b = Batch(b=np.zeros([4, 3]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.cat([a, b])
            >>> c.a.shape
            (7, 4)
            >>> c.b.shape
            (7, 3)
            >>> c.common.c.shape
            (7, 5)
        """
        ...

    def stack_(self, batches: Sequence[dict | Self], axis: int = 0) -> None:
        """Stack a list of Batch object into current batch."""
        ...

    @staticmethod
    def stack(batches: Sequence[dict | TBatch], axis: int = 0) -> TBatch:
        """Stack a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros. E.g.
        ::

            >>> a = Batch(a=np.zeros([4, 4]), common=Batch(c=np.zeros([4, 5])))
            >>> b = Batch(b=np.zeros([4, 6]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.stack([a, b])
            >>> c.a.shape
            (2, 4, 4)
            >>> c.b.shape
            (2, 4, 6)
            >>> c.common.c.shape
            (2, 4, 5)

        .. note::

            If there are keys that are not shared across all batches, ``stack``
            with ``axis != 0`` is undefined, and will cause an exception.
        """
        ...

    def empty_(self, index: slice | IndexType | None = None) -> Self:
        """Return an empty Batch object with 0 or None filled.

        If "index" is specified, it will only reset the specific indexed-data.
        ::

            >>> data.empty_()
            >>> print(data)
            Batch(
                a: array([[0., 0.],
                          [0., 0.]]),
                b: array([None, None], dtype=object),
            )
            >>> b={'c': [2., 'st'], 'd': [1., 0.]}
            >>> data = Batch(a=[False,  True], b=b)
            >>> data[0] = Batch.empty(data[1])
            >>> data
            Batch(
                a: array([False,  True]),
                b: Batch(
                       c: array([None, 'st']),
                       d: array([0., 0.]),
                   ),
            )
        """
        ...

    @staticmethod
    def empty(batch: TBatch, index: IndexType | None = None) -> TBatch:
        """Return an empty Batch object with 0 or None filled.

        The shape is the same as the given Batch.
        """
        ...

    def update(self, batch: dict | Self | None = None, **kwargs: Any) -> None:
        """Update this batch from another dict/Batch."""
        ...

    def __len__(self) -> int:
        ...

    def split(
        self,
        size: int,
        shuffle: bool = True,
        merge_last: bool = False,
    ) -> Iterator[Self]:
        """Split whole data into multiple small batches.

        :param size: divide the data batch with the given size, but one
            batch if the length of the batch is smaller than "size". Size of -1 means
            the whole batch.
        :param shuffle: randomly shuffle the entire data batch if it is
            True, otherwise remain in the same. Default to True.
        :param merge_last: merge the last batch into the previous one.
            Default to False.
        """
        ...

    def to_dict(self, recurse: bool = True) -> dict[str, Any]:
        ...

    def to_list_of_dicts(self) -> list[dict[str, Any]]:
        ...


class Batch(BatchProtocol):
    """See :class:`~tianshou.data.batch.BatchProtocol`."""

    __doc__ = BatchProtocol.__doc__

    def __init__(
        self,
        batch_dict: dict
        | BatchProtocol
        | Sequence[dict | BatchProtocol]
        | np.ndarray
        | None = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> None:
        if copy:
            batch_dict = deepcopy(batch_dict)
        if batch_dict is not None:
            if isinstance(batch_dict, dict | BatchProtocol):
                _assert_type_keys(batch_dict.keys())
                for batch_key, obj in batch_dict.items():
                    self.__dict__[batch_key] = _parse_value(obj)
            elif _is_batch_set(batch_dict):
                batch_dict = cast(Sequence[dict | BatchProtocol], batch_dict)
                self.stack_(batch_dict)
        if len(kwargs) > 0:
            # TODO: that's a rather weird pattern, is it really needed?
            # Feels like kwargs could be just merged into batch_dict in the beginning
            self.__init__(kwargs, copy=copy)  # type: ignore

    def to_dict(self, recursive: bool = True) -> dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if recursive and isinstance(v, Batch):
                v = v.to_dict(recursive=recursive)
            result[k] = v
        return result

    def get_keys(self) -> KeysView:
        return self.__dict__.keys()

    def to_list_of_dicts(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self]

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        self.__dict__[key] = _parse_value(value)

    def __getattr__(self, key: str) -> Any:
        """Return self.key. The "Any" return type is needed for mypy."""
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) -> bool:
        """Return key in self."""
        return key in self.__dict__

    def __getstate__(self) -> dict[str, Any]:
        """Pickling interface.

        Only the actual data are serialized for both efficiency and simplicity.
        """
        state = {}
        for batch_key, obj in self.items():
            if isinstance(obj, Batch):
                state[batch_key] = obj.__getstate__()
            else:
                state[batch_key] = obj
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickling interface.

        At this point, self is an empty Batch instance that has not been
        initialized, so it can safely be initialized by the pickle state.
        """
        self.__init__(**state)  # type: ignore

    @overload
    def __getitem__(self, index: str) -> Any:
        ...

    @overload
    def __getitem__(self, index: IndexType) -> Self:
        ...

    def __getitem__(self, index: str | IndexType) -> Any:
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch_items = self.items()
        if len(batch_items) > 0:
            new_batch = Batch()
            for batch_key, obj in batch_items:
                if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                    new_batch.__dict__[batch_key] = Batch()
                else:
                    new_batch.__dict__[batch_key] = obj[index]
            return new_batch
        raise IndexError("Cannot access item from empty Batch object.")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        this_batch_no_torch_tensor: Batch = Batch.to_numpy(self)
        other_batch_no_torch_tensor: Batch = Batch.to_numpy(other)
        this_dict = this_batch_no_torch_tensor.to_dict(recursive=True)
        other_dict = other_batch_no_torch_tensor.to_dict(recursive=True)

        return not DeepDiff(this_dict, other_dict)

    def __iter__(self) -> Iterator[Self]:
        # TODO: empty batch raises an error on len and needs separate treatment, that's probably not a good idea
        if len(self.__dict__) == 0:
            yield from []
        else:
            for i in range(len(self)):
                yield self[i]

    def __setitem__(self, index: str | IndexType, value: Any) -> None:
        """Assign value to self[index]."""
        value = _parse_value(value)
        if isinstance(index, str):
            self.__dict__[index] = value
            return
        if not isinstance(value, Batch):
            raise ValueError(
                "Batch does not supported tensor assignment. "
                "Use a compatible Batch or dict instead.",
            )
        if not set(value.keys()).issubset(self.__dict__.keys()):
            raise ValueError("Creating keys is not supported by item assignment.")
        for key, val in self.items():
            try:
                self.__dict__[key][index] = value[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][index] = Batch()
                elif isinstance(val, torch.Tensor) or (
                    isinstance(val, np.ndarray) and issubclass(val.dtype.type, np.bool_ | np.number)
                ):
                    self.__dict__[key][index] = 0
                else:
                    self.__dict__[key][index] = None

    def __iadd__(self, other: Self | Number | np.number) -> Self:
        """Algebraic addition with another Batch instance in-place."""
        if isinstance(other, Batch):
            for (batch_key, obj), value in zip(
                self.__dict__.items(),
                other.__dict__.values(),
                strict=True,
            ):  # TODO are keys consistent?
                if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                    continue
                self.__dict__[batch_key] += value
            return self
        if _is_number(other):
            for batch_key, obj in self.items():
                if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                    continue
                self.__dict__[batch_key] += other
            return self
        raise TypeError("Only addition of Batch or number is supported.")

    def __add__(self, other: Self | Number | np.number) -> Self:
        """Algebraic addition with another Batch instance out-of-place."""
        return deepcopy(self).__iadd__(other)

    def __imul__(self, value: Number | np.number) -> Self:
        """Algebraic multiplication with a scalar value in-place."""
        assert _is_number(value), "Only multiplication by a number is supported."
        for batch_key, obj in self.__dict__.items():
            if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                continue
            self.__dict__[batch_key] *= value
        return self

    def __mul__(self, value: Number | np.number) -> Self:
        """Algebraic multiplication with a scalar value out-of-place."""
        return deepcopy(self).__imul__(value)

    def __itruediv__(self, value: Number | np.number) -> Self:
        """Algebraic division with a scalar value in-place."""
        assert _is_number(value), "Only division by a number is supported."
        for batch_key, obj in self.__dict__.items():
            if isinstance(obj, Batch) and len(obj.get_keys()) == 0:
                continue
            self.__dict__[batch_key] /= value
        return self

    def __truediv__(self, value: Number | np.number) -> Self:
        """Algebraic division with a scalar value out-of-place."""
        return deepcopy(self).__itruediv__(value)

    def __repr__(self) -> str:
        """Return str(self)."""
        self_str = self.__class__.__name__ + "(\n"
        flag = False
        for batch_key, obj in self.__dict__.items():
            rpl = "\n" + " " * (6 + len(batch_key))
            obj_name = pprint.pformat(obj).replace("\n", rpl)
            self_str += f"    {batch_key}: {obj_name},\n"
            flag = True
        if flag:
            self_str += ")"
        else:
            self_str = self.__class__.__name__ + "()"
        return self_str

    @staticmethod
    def to_numpy(batch: TBatch) -> TBatch:
        batch_dict = deepcopy(batch)
        for batch_key, obj in batch_dict.items():
            if isinstance(obj, torch.Tensor):
                batch_dict.__dict__[batch_key] = obj.detach().cpu().numpy()
            elif isinstance(obj, Batch):
                obj = Batch.to_numpy(obj)
                batch_dict.__dict__[batch_key] = obj

        return batch_dict

    def to_numpy_(self) -> None:
        for batch_key, obj in self.items():
            if isinstance(obj, torch.Tensor):
                self.__dict__[batch_key] = obj.detach().cpu().numpy()
            elif isinstance(obj, Batch):
                obj.to_numpy_()

    @staticmethod
    def to_torch(
        batch: TBatch,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> TBatch:
        new_batch = Batch(batch, copy=True)
        new_batch.to_torch_(dtype=dtype, device=device)

        return new_batch  # type: ignore[return-value]

    def to_torch_(
        self,
        dtype: torch.dtype | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        if not isinstance(device, torch.device):
            device = torch.device(device)

        for batch_key, obj in self.items():
            if isinstance(obj, torch.Tensor):
                if (
                    dtype is not None
                    and obj.dtype != dtype
                    or obj.device.type != device.type
                    or device.index != obj.device.index
                ):
                    if dtype is not None:
                        self.__dict__[batch_key] = obj.type(dtype).to(device)
                    else:
                        self.__dict__[batch_key] = obj.to(device)
            elif isinstance(obj, Batch):
                obj.to_torch_(dtype, device)
            else:
                # ndarray or scalar
                if not isinstance(obj, np.ndarray):
                    obj = np.asanyarray(obj)
                obj = torch.from_numpy(obj).to(device)
                if dtype is not None:
                    obj = obj.type(dtype)
                self.__dict__[batch_key] = obj

    def __cat(self, batches: Sequence[dict | Self], lens: list[int]) -> None:
        """Private method for Batch.cat_.

        ::

            >>> a = Batch(a=np.random.randn(3, 4))
            >>> x = Batch(a=a, b=np.random.randn(4, 4))
            >>> y = Batch(a=Batch(a=Batch()), b=np.random.randn(4, 4))

        If we want to concatenate x and y, we want to pad y.a.a with zeros.
        Without ``lens`` as a hint, when we concatenate x.a and y.a, we would
        not be able to know how to pad y.a. So ``Batch.cat_`` should compute
        the ``lens`` to give ``Batch.__cat`` a hint.
        ::

            >>> ans = Batch.cat([x, y])
            >>> # this is equivalent to the following line
            >>> ans = Batch(); ans.__cat([x, y], lens=[3, 4])
            >>> # this lens is equal to [len(a), len(b)]
        """
        # partial keys will be padded by zeros
        # with the shape of [len, rest_shape]
        sum_lens = [0]
        for len_ in lens:
            sum_lens.append(sum_lens[-1] + len_)
        # collect non-empty keys
        keys_map = [
            {
                batch_key
                for batch_key, obj in batch.items()
                if not (isinstance(obj, Batch) and len(obj.get_keys()) == 0)
            }
            for batch in batches
        ]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[batch[key] for batch in batches] for key in keys_shared]
        for key, shared_value in zip(keys_shared, values_shared, strict=True):
            if all(isinstance(element, dict | Batch) for element in shared_value):
                batch_holder = Batch()
                batch_holder.__cat(shared_value, lens=lens)
                self.__dict__[key] = batch_holder
            elif all(isinstance(element, torch.Tensor) for element in shared_value):
                self.__dict__[key] = torch.cat(shared_value)
            else:
                # cat Batch(a=np.zeros((3, 4))) and Batch(a=Batch(b=Batch()))
                # will fail here
                self.__dict__[key] = _to_array_with_correct_type(np.concatenate(shared_value))
        keys_total = set.union(*[set(batch.keys()) for batch in batches])
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        for key in keys_reserve:
            # reserved keys
            self.__dict__[key] = Batch()
        for key in keys_partial:
            for i, batch in enumerate(batches):
                if key not in batch.__dict__:
                    continue
                value = batch.get(key)
                if isinstance(value, Batch) and len(value.get_keys()) == 0:
                    continue
                try:
                    self.__dict__[key][sum_lens[i] : sum_lens[i + 1]] = value
                except KeyError:
                    self.__dict__[key] = create_value(value, sum_lens[-1], stack=False)
                    self.__dict__[key][sum_lens[i] : sum_lens[i + 1]] = value

    def cat_(self, batches: BatchProtocol | Sequence[dict | BatchProtocol]) -> None:
        if isinstance(batches, BatchProtocol | dict):
            batches = [batches]
        # check input format
        batch_list = []
        for batch in batches:
            if isinstance(batch, dict):
                if len(batch) > 0:
                    batch_list.append(Batch(batch))
            elif isinstance(batch, Batch):
                if len(batch.get_keys()) != 0:
                    batch_list.append(batch)
            else:
                raise ValueError(f"Cannot concatenate {type(batch)} in Batch.cat_")
        if len(batch_list) == 0:
            return
        batches = batch_list
        try:
            # len(batch) here means batch is a nested empty batch
            # like Batch(a=Batch), and we have to treat it as length zero and
            # keep it.
            lens = [0 if len(batch) == 0 else len(batch) for batch in batches]
        except TypeError as exception:
            raise ValueError(
                "Batch.cat_ meets an exception. Maybe because there is any "
                f"scalar in {batches} but Batch.cat_ does not support the "
                "concatenation of scalar.",
            ) from exception
        if len(self.get_keys()) != 0:
            batches = [self, *list(batches)]
            lens = [0 if len(self) == 0 else len(self), *lens]
        self.__cat(batches, lens)

    @staticmethod
    def cat(batches: Sequence[dict | TBatch]) -> TBatch:
        batch = Batch()
        batch.cat_(batches)
        return batch  # type: ignore

    def stack_(self, batches: Sequence[dict | BatchProtocol], axis: int = 0) -> None:
        # check input format
        batch_list = []
        for batch in batches:
            if isinstance(batch, dict):
                if len(batch) > 0:
                    batch_list.append(Batch(batch))
            elif isinstance(batch, Batch):
                if len(batch.get_keys()) != 0:
                    batch_list.append(batch)
            else:
                raise ValueError(f"Cannot concatenate {type(batch)} in Batch.stack_")
        if len(batch_list) == 0:
            return
        batches = batch_list
        if len(self.get_keys()) != 0:
            batches = [self, *batches]
        # collect non-empty keys
        keys_map = [
            {
                batch_key
                for batch_key, obj in batch.items()
                if not (isinstance(obj, BatchProtocol) and len(obj.get_keys()) == 0)
            }
            for batch in batches
        ]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[batch[key] for batch in batches] for key in keys_shared]
        for shared_key, value in zip(keys_shared, values_shared, strict=True):
            # second often
            if all(isinstance(element, torch.Tensor) for element in value):
                self.__dict__[shared_key] = torch.stack(value, axis)
            # third often
            elif all(isinstance(element, BatchProtocol | dict) for element in value):
                self.__dict__[shared_key] = Batch.stack(value, axis)
            else:  # most often case is np.ndarray
                try:
                    self.__dict__[shared_key] = _to_array_with_correct_type(np.stack(value, axis))
                except ValueError:
                    warnings.warn(
                        "You are using tensors with different shape,"
                        " fallback to dtype=object by default.",
                    )
                    self.__dict__[shared_key] = np.array(value, dtype=object)
        # all the keys
        keys_total = set.union(*[set(batch.keys()) for batch in batches])
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that are either partial or reserved
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        if keys_partial and axis != 0:
            raise ValueError(
                f"Stack of Batch with non-shared keys {keys_partial} is only "
                f"supported with axis=0, but got axis={axis}!",
            )
        for key in keys_reserve:
            # reserved keys
            self.__dict__[key] = Batch()
        for key in keys_partial:
            for i, batch in enumerate(batches):
                if key not in batch.__dict__:
                    continue
                value = batch.get(key)
                # TODO: fix code/annotations s.t. the ignores can be removed
                if (
                    isinstance(value, BatchProtocol)  # type: ignore
                    and len(value.get_keys()) == 0  # type: ignore
                ):
                    continue  # type: ignore
                try:
                    self.__dict__[key][i] = value
                except KeyError:
                    self.__dict__[key] = create_value(value, len(batches))
                    self.__dict__[key][i] = value

    @staticmethod
    def stack(batches: Sequence[dict | TBatch], axis: int = 0) -> TBatch:
        batch = Batch()
        batch.stack_(batches, axis)
        # can't cast to a generic type, so we have to ignore the type here
        return batch  # type: ignore

    def empty_(self, index: slice | IndexType | None = None) -> Self:
        for batch_key, obj in self.items():
            if isinstance(obj, torch.Tensor):  # most often case
                self.__dict__[batch_key][index] = 0
            elif obj is None:
                continue
            elif isinstance(obj, np.ndarray):
                if obj.dtype == object:
                    self.__dict__[batch_key][index] = None
                else:
                    self.__dict__[batch_key][index] = 0
            elif isinstance(obj, Batch):
                self.__dict__[batch_key].empty_(index=index)
            else:  # scalar value
                warnings.warn(
                    "You are calling Batch.empty on a NumPy scalar, "
                    "which may cause undefined behaviors.",
                )
                if _is_number(obj):
                    self.__dict__[batch_key] = obj.__class__(0)
                else:
                    self.__dict__[batch_key] = None
        return self

    @staticmethod
    def empty(batch: TBatch, index: IndexType | None = None) -> TBatch:
        return deepcopy(batch).empty_(index)

    def update(self, batch: dict | Self | None = None, **kwargs: Any) -> None:
        if batch is None:
            self.update(kwargs)
            return
        for batch_key, obj in batch.items():
            self.__dict__[batch_key] = _parse_value(obj)
        if kwargs:
            self.update(kwargs)

    def __len__(self) -> int:
        """Return len(self)."""
        lens = []
        for obj in self.__dict__.values():
            # TODO: causes inconsistent behavior to batch with empty batches
            #  and batch with empty sequences of other type. Remove, but only after
            #  Buffer and Collectors have been improved to no longer rely on this
            if isinstance(obj, Batch) and len(obj) == 0:
                continue
            if hasattr(obj, "__len__") and (isinstance(obj, Batch) or obj.ndim > 0):
                lens.append(len(obj))
            else:
                raise TypeError(f"Object {obj} in {self} has no len()")
        if not lens:
            return 0
        return min(lens)

    @property
    def shape(self) -> list[int]:
        """Return self.shape."""
        if len(self.get_keys()) == 0:
            return []
        data_shape = []
        for obj in self.__dict__.values():
            try:
                data_shape.append(list(obj.shape))
            except AttributeError:
                data_shape.append([])
        return (
            list(map(min, zip(*data_shape, strict=False))) if len(data_shape) > 1 else data_shape[0]
        )

    def split(
        self,
        size: int,
        shuffle: bool = True,
        merge_last: bool = False,
    ) -> Iterator[Self]:
        length = len(self)
        if size == -1:
            size = length
        assert size >= 1  # size can be greater than length, return whole batch
        indices = np.random.permutation(length) if shuffle else np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield self[indices[idx:]]
                break
            yield self[indices[idx : idx + size]]
