import copy
import pickle
import sys
from itertools import starmap
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
import torch
from deepdiff import DeepDiff

from tianshou.data import Batch, to_numpy, to_torch


def test_batch() -> None:
    assert list(Batch()) == []
    assert len(Batch().get_keys()) == 0
    assert len(Batch(b={"c": {}}).get_keys()) != 0
    assert len(Batch(b={"c": {}})) == 0
    assert len(Batch(a=Batch(), b=Batch(c=Batch())).get_keys()) != 0
    assert len(Batch(a=Batch(), b=Batch(c=Batch()))) == 0
    assert len(Batch(d=1).get_keys()) != 0
    assert len(Batch(a=np.float64(1.0)).get_keys()) != 0
    assert len(Batch(a=[1, 2, 3], b={"c": {}})) == 3
    assert len(Batch(a=[1, 2, 3]).get_keys()) != 0
    b = Batch({"a": [4, 4], "b": [5, 5]}, c=[None, None])
    assert b.c.dtype == object
    b = Batch(d=[None], e=[starmap], f=Batch)
    assert b.d.dtype == b.e.dtype == object
    assert b.f == Batch
    b = Batch()
    b.update()
    assert len(b.get_keys()) == 0
    b.update(c=[3, 5])
    assert np.allclose(b.c, [3, 5])
    # mimic the behavior of dict.update, where kwargs can overwrite keys
    b.update({"a": 2}, a=3)
    assert "a" in b
    assert b.a == 3
    assert b.pop("a") == 3
    assert "a" not in b
    with pytest.raises(AssertionError):
        Batch({1: 2})
    batch = Batch(a=[torch.ones(3), torch.ones(3)])
    assert Batch(a=[np.zeros((2, 3)), np.zeros((3, 3))]).a.dtype == object
    with pytest.raises(TypeError):
        Batch(a=[np.zeros((3, 2)), np.zeros((3, 3))])
    with pytest.raises(TypeError):
        Batch(a=[torch.zeros((2, 3)), torch.zeros((3, 3))])
    with pytest.raises(TypeError):
        Batch(a=[torch.zeros((3, 3)), np.zeros((3, 3))])
    with pytest.raises(TypeError):
        Batch(a=[1, np.zeros((3, 3)), torch.zeros((3, 3))])
    assert torch.allclose(batch.a, torch.ones(2, 3))
    batch.cat_(batch)
    assert torch.allclose(batch.a, torch.ones(4, 3))
    Batch(a=[])
    batch = Batch(obs=[0], np=np.zeros([3, 4]))
    assert batch.obs == batch["obs"]
    batch.obs = [1]
    assert batch.obs == [1]
    batch.cat_(batch)
    assert np.allclose(batch.obs, [1, 1])
    assert batch.np.shape == (6, 4)
    assert np.allclose(batch[0].obs, batch[1].obs)
    batch.obs = np.arange(5)
    for i, b in enumerate(batch.split(1, shuffle=False)):
        if i != 5:
            assert b.obs == batch[i].obs
        else:
            with pytest.raises(AttributeError):
                batch[i].obs  # noqa: B018
            with pytest.raises(AttributeError):
                b.obs  # noqa: B018
    print(batch)
    batch = Batch(a=np.arange(10))
    with pytest.raises(AssertionError):
        list(batch.split(0))
    data = [
        (1, False, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]),
        (1, True, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]),
        (3, False, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]),
        (3, True, [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]),
        (5, False, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        (5, True, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        (7, False, [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]]),
        (7, True, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (10, False, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (10, True, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (15, False, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (15, True, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (100, False, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (100, True, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    ]
    for size, merge_last, result in data:
        bs = list(batch.split(size, shuffle=False, merge_last=merge_last))
        assert [bs[i].a.tolist() for i in range(len(bs))] == result
    batch_dict = {"b": np.array([1.0]), "c": 2.0, "d": torch.Tensor([3.0])}
    batch_item = Batch({"a": [batch_dict]})[0]
    assert isinstance(batch_item.a.b, np.ndarray)
    assert batch_item.a.b == batch_dict["b"]
    assert isinstance(batch_item.a.c, float)
    assert batch_item.a.c == batch_dict["c"]
    assert isinstance(batch_item.a.d, torch.Tensor)
    assert batch_item.a.d == batch_dict["d"]
    batch2 = Batch(a=[{"b": np.float64(1.0), "c": np.zeros(1), "d": Batch(e=np.array(3.0))}])
    assert len(batch2) == 1
    assert Batch().shape == []
    assert Batch(a=1).shape == []
    assert Batch(a={1, 2}).shape == []
    assert batch2.shape[0] == 1
    assert "a" in batch2
    assert all(i in batch2.a for i in "bcd")
    with pytest.raises(IndexError):
        batch2[-2]
    with pytest.raises(IndexError):
        batch2[1]
    assert batch2[0].shape == []
    with pytest.raises(IndexError):
        batch2[0][0]
    with pytest.raises(TypeError):
        len(batch2[0])
    assert isinstance(batch2[0].a.c, np.ndarray)
    assert isinstance(batch2[0].a.b, np.float64)
    assert isinstance(batch2[0].a.d.e, np.float64)
    batch2_from_list = Batch(list(batch2))
    batch2_from_comp = Batch(list(batch2))
    assert batch2_from_list.a.b == batch2.a.b
    assert batch2_from_list.a.c == batch2.a.c
    assert batch2_from_list.a.d.e == batch2.a.d.e
    assert batch2_from_comp.a.b == batch2.a.b
    assert batch2_from_comp.a.c == batch2.a.c
    assert batch2_from_comp.a.d.e == batch2.a.d.e
    for batch_slice in [batch2[slice(0, 1)], batch2[:1], batch2[0:]]:
        assert batch_slice.a.b == batch2.a.b
        assert batch_slice.a.c == batch2.a.c
        assert batch_slice.a.d.e == batch2.a.d.e
    batch2.a.d.f = {}
    batch2_sum = (batch2 + 1.0) * 2  # type: ignore  # __add__ supports Number as input type
    assert batch2_sum.a.b == (batch2.a.b + 1.0) * 2
    assert batch2_sum.a.c == (batch2.a.c + 1.0) * 2
    assert batch2_sum.a.d.e == (batch2.a.d.e + 1.0) * 2
    assert len(batch2_sum.a.d.f.get_keys()) == 0
    with pytest.raises(TypeError):
        batch2 += [1]  # type: ignore  # error is raised explicitly
    batch3 = Batch(a={"c": np.zeros(1), "d": Batch(e=np.array([0.0]), f=np.array([3.0]))})
    batch3.a.d[0] = {"e": 4.0}
    assert batch3.a.d.e[0] == 4.0
    batch3.a.d[0] = Batch(f=5.0)
    assert batch3.a.d.f[0] == 5.0
    with pytest.raises(ValueError):
        batch3.a.d[0] = Batch(f=5.0, g=0.0)
    with pytest.raises(ValueError):
        batch3[0] = Batch(a={"c": 2, "e": 1})
    # auto convert
    batch4 = Batch(a=np.array(["a", "b"]))
    assert batch4.a.dtype == object  # auto convert to object
    batch4.update(a=np.array(["c", "d"]))
    assert list(batch4.a) == ["c", "d"]
    assert batch4.a.dtype == object  # auto convert to object
    batch5 = Batch(a=np.array([{"index": 0}]))
    assert isinstance(batch5.a, Batch)
    assert np.allclose(batch5.a.index, [0])
    # We use setattr b/c the setattr of Batch will actually change the type of the field that is being set!
    # However, mypy would not understand this, and rightly expect that batch.b = some_array would lead to
    # batch.b being an array (which it is not, it's turned into a Batch instead)
    batch5.b = np.array([{"index": 1}])
    batch5.b = cast(Batch, batch5.b)
    assert isinstance(batch5.b, Batch)
    assert np.allclose(batch5.b.index, [1])

    # None is a valid object and can be stored in Batch
    a = Batch.stack([Batch(a=None), Batch(b=None)])
    assert a.a[0] is None
    assert a.a[1] is None
    assert a.b[0] is None
    assert a.b[1] is None

    # nx.Graph corner case
    assert Batch(a=np.array([nx.Graph(), nx.Graph()], dtype=object)).a.dtype == object
    g1 = nx.Graph()
    g1.add_nodes_from(list(range(10)))
    g2 = nx.Graph()
    g2.add_nodes_from(list(range(20)))
    assert Batch(a=np.array([g1, g2], dtype=object)).a.dtype == object


def test_batch_over_batch() -> None:
    batch = Batch(a=[3, 4, 5], b=[4, 5, 6])
    batch2 = Batch({"c": [6, 7, 8], "b": batch})
    batch2.b.b[-1] = 0
    print(batch2)
    for k, v in batch2.items():
        assert np.all(batch2[k] == v)
    assert batch2[-1].b.b == 0
    batch2.cat_(Batch(c=[6, 7, 8], b=batch))
    assert np.allclose(batch2.c, [6, 7, 8, 6, 7, 8])
    assert np.allclose(batch2.b.a, [3, 4, 5, 3, 4, 5])
    assert np.allclose(batch2.b.b, [4, 5, 0, 4, 5, 0])
    batch2.update(batch2.b, six=[6, 6, 6])
    assert np.allclose(batch2.c, [6, 7, 8, 6, 7, 8])
    assert np.allclose(batch2.a, [3, 4, 5, 3, 4, 5])
    assert np.allclose(batch2.b, [4, 5, 0, 4, 5, 0])
    assert np.allclose(batch2.six, [6, 6, 6])
    d = {"a": [3, 4, 5], "b": [4, 5, 6]}
    batch3 = Batch(c=[6, 7, 8], b=d)
    batch3.cat_(Batch(c=[6, 7, 8], b=d))
    assert np.allclose(batch3.c, [6, 7, 8, 6, 7, 8])
    assert np.allclose(batch3.b.a, [3, 4, 5, 3, 4, 5])
    assert np.allclose(batch3.b.b, [4, 5, 6, 4, 5, 6])
    batch4 = Batch(({"a": {"b": np.array([1.0])}},))
    assert batch4.a.b.ndim == 2
    assert batch4.a.b[0, 0] == 1.0
    # advanced slicing
    batch5 = Batch(a=[[1, 2]], b={"c": np.zeros([3, 2, 1])})
    assert batch5.shape == [1, 2]
    with pytest.raises(IndexError):
        batch5[2]
    with pytest.raises(IndexError):
        batch5[:, 3]
    with pytest.raises(IndexError):
        batch5[:, :, -1]
    batch5[:, -1] += np.int_(1)
    assert np.allclose(batch5.a, [1, 3])
    assert np.allclose(batch5.b.c.squeeze(), [[0, 1]] * 3)
    with pytest.raises(ValueError):
        batch5[:, -1] = 1
    batch5[:, 0] = {"a": -1}
    assert np.allclose(batch5.a, [-1, 3])
    assert np.allclose(batch5.b.c.squeeze(), [[0, 1]] * 3)


def test_batch_cat_and_stack() -> None:
    # test cat with compatible keys
    b1 = Batch(a=[{"b": np.float64(1.0), "d": Batch(e=np.array(3.0))}])
    b2 = Batch(a=[{"b": np.float64(4.0), "d": {"e": np.array(6.0)}}])
    b12_cat_out = Batch.cat([b1, b2])
    b12_cat_in = copy.deepcopy(b1)
    b12_cat_in.cat_(b2)
    assert np.all(b12_cat_in.a.d.e == b12_cat_out.a.d.e)
    assert np.all(b12_cat_in.a.d.e == b12_cat_out.a.d.e)
    assert isinstance(b12_cat_in.a.d.e, np.ndarray)
    assert b12_cat_in.a.d.e.ndim == 1

    a = Batch(a=Batch(a=np.random.randn(3, 4)))
    assert np.allclose(
        np.concatenate([a.a.a, a.a.a]),
        Batch.cat([a, Batch(a=Batch(a=Batch())), a]).a.a,
    )

    # test cat with lens infer
    a = Batch(a=Batch(a=np.random.randn(3, 4)), b=np.random.randn(3, 4))
    b = Batch(a=Batch(a=Batch(), t=Batch()), b=np.random.randn(3, 4))
    ans = Batch.cat([a, b, a])
    assert np.allclose(ans.a.a, np.concatenate([a.a.a, np.zeros((3, 4)), a.a.a]))
    assert np.allclose(ans.b, np.concatenate([a.b, b.b, a.b]))
    assert len(ans.a.t.get_keys()) == 0

    b1.stack_([b2])
    assert isinstance(b1.a.d.e, np.ndarray)
    assert b1.a.d.e.ndim == 2

    # test cat with incompatible keys
    b1 = Batch(a=np.random.rand(3, 4), common=Batch(c=np.random.rand(3, 5)))
    b2 = Batch(b=torch.rand(4, 3), common=Batch(c=np.random.rand(4, 5)))
    test = Batch.cat([b1, b2])
    ans = Batch(
        a=np.concatenate([b1.a, np.zeros((4, 4))]),
        b=torch.cat([torch.zeros(3, 3), b2.b]),
        common=Batch(c=np.concatenate([b1.common.c, b2.common.c])),
    )
    assert np.allclose(test.a, ans.a)
    assert torch.allclose(test.b, ans.b)
    assert np.allclose(test.common.c, ans.common.c)

    # test cat with reserved keys (values are Batch())
    b1 = Batch(a=np.random.rand(3, 4), common=Batch(c=np.random.rand(3, 5)))
    b2 = Batch(a=Batch(), b=torch.rand(4, 3), common=Batch(c=np.random.rand(4, 5)))
    test = Batch.cat([b1, b2])
    ans = Batch(
        a=np.concatenate([b1.a, np.zeros((4, 4))]),
        b=torch.cat([torch.zeros(3, 3), b2.b]),
        common=Batch(c=np.concatenate([b1.common.c, b2.common.c])),
    )
    assert np.allclose(test.a, ans.a)
    assert torch.allclose(test.b, ans.b)
    assert np.allclose(test.common.c, ans.common.c)

    # test cat with all reserved keys (values are Batch())
    b1 = Batch(a=Batch(), common=Batch(c=np.random.rand(3, 5)))
    b2 = Batch(a=Batch(), b=torch.rand(4, 3), common=Batch(c=np.random.rand(4, 5)))
    test = Batch.cat([b1, b2])
    ans = Batch(
        a=Batch(),
        b=torch.cat([torch.zeros(3, 3), b2.b]),
        common=Batch(c=np.concatenate([b1.common.c, b2.common.c])),
    )
    assert len(ans.a.get_keys()) == 0
    assert torch.allclose(test.b, ans.b)
    assert np.allclose(test.common.c, ans.common.c)

    # test stack with compatible keys
    b3 = Batch(a=np.zeros((3, 4)), b=torch.ones((2, 5)), c=Batch(d=[[1], [2]]))
    b4 = Batch(a=np.ones((3, 4)), b=torch.ones((2, 5)), c=Batch(d=[[0], [3]]))
    b34_stack = Batch.stack((b3, b4), axis=1)
    assert np.all(b34_stack.a == np.stack((b3.a, b4.a), axis=1))
    assert np.all(b34_stack.c.d == list(map(list, zip(b3.c.d, b4.c.d, strict=True))))
    b5_dict = np.array([{"a": False, "b": {"c": 2.0, "d": 1.0}}, {"a": True, "b": {"c": 3.0}}])
    b5 = Batch(b5_dict)
    assert b5.a[0] == np.array(False)
    assert b5.a[1] == np.array(True)
    assert np.all(b5.b.c == np.stack([e["b"]["c"] for e in b5_dict], axis=0))
    assert b5.b.d[0] == b5_dict[0]["b"]["d"]
    assert b5.b.d[1] == 0.0

    # test stack with incompatible keys
    a = Batch(a=1, b=2, c=3)
    b = Batch(a=4, b=5, d=6)
    c = Batch(c=7, b=6, d=9)
    d = Batch.stack([a, b, c])
    assert np.allclose(d.a, [1, 4, 0])
    assert np.allclose(d.b, [2, 5, 6])
    assert np.allclose(d.c, [3, 0, 7])
    assert np.allclose(d.d, [0, 6, 9])

    # test stack with empty Batch()
    assert len(Batch.stack([Batch(), Batch(), Batch()]).get_keys()) == 0
    a = Batch(a=1, b=2, c=3, d=Batch(), e=Batch())
    b = Batch(a=4, b=5, d=6, e=Batch())
    c = Batch(c=7, b=6, d=9, e=Batch())
    d = Batch.stack([a, b, c])
    assert np.allclose(d.a, [1, 4, 0])
    assert np.allclose(d.b, [2, 5, 6])
    assert np.allclose(d.c, [3, 0, 7])
    assert np.allclose(d.d, [0, 6, 9])
    assert len(d.e.get_keys()) == 0
    b1 = Batch(a=Batch(), common=Batch(c=np.random.rand(4, 5)))
    b2 = Batch(b=Batch(), common=Batch(c=np.random.rand(4, 5)))
    test = Batch.stack([b1, b2], axis=-1)
    assert len(test.a.get_keys()) == 0
    assert len(test.b.get_keys()) == 0
    assert np.allclose(test.common.c, np.stack([b1.common.c, b2.common.c], axis=-1))

    b1 = Batch(a=np.random.rand(4, 4), common=Batch(c=np.random.rand(4, 5)))
    b2 = Batch(b=torch.rand(4, 6), common=Batch(c=np.random.rand(4, 5)))
    test = Batch.stack([b1, b2])
    ans = Batch(
        a=np.stack([b1.a, np.zeros((4, 4))]),
        b=torch.stack([torch.zeros(4, 6), b2.b]),
        common=Batch(c=np.stack([b1.common.c, b2.common.c])),
    )
    assert np.allclose(test.a, ans.a)
    assert torch.allclose(test.b, ans.b)
    assert np.allclose(test.common.c, ans.common.c)

    # test with illegal input format
    with pytest.raises(ValueError):
        Batch.cat([[Batch(a=1)], [Batch(a=1)]])  # type: ignore  # cat() tested with invalid inp
    with pytest.raises(ValueError):
        Batch.stack([[Batch(a=1)], [Batch(a=1)]])  # type: ignore # stack() tested with invalid inp

    # exceptions
    batch_cat: Batch = Batch.cat([])
    assert len(batch_cat.get_keys()) == 0
    batch_stack: Batch = Batch.stack([])
    assert len(batch_stack.get_keys()) == 0
    b1 = Batch(e=[4, 5], d=6)
    b2 = Batch(e=[4, 6])
    with pytest.raises(ValueError):
        Batch.cat([b1, b2])
    with pytest.raises(ValueError):
        Batch.stack([b1, b2], axis=1)


def test_batch_over_batch_to_torch() -> None:
    batch = Batch(
        a=np.float64(1.0),
        b=Batch(c=np.ones((1,), dtype=np.float32), d=torch.ones((1,), dtype=torch.float64)),
    )
    batch.b.__dict__["e"] = 1  # bypass the check
    batch.to_torch_()
    assert isinstance(batch.a, torch.Tensor)
    assert isinstance(batch.b.c, torch.Tensor)
    assert isinstance(batch.b.d, torch.Tensor)
    assert isinstance(batch.b.e, torch.Tensor)
    assert batch.a.dtype == torch.float64
    assert batch.b.c.dtype == torch.float32
    assert batch.b.d.dtype == torch.float64
    if sys.platform in ["win32", "cygwin"]:  # windows
        assert batch.b.e.dtype == torch.int32
    else:
        assert batch.b.e.dtype == torch.int64
    batch.to_torch_(dtype=torch.float32)
    assert batch.a.dtype == torch.float32
    assert batch.b.c.dtype == torch.float32
    assert batch.b.d.dtype == torch.float32
    assert batch.b.e.dtype == torch.float32


def test_utils_to_torch_numpy() -> None:
    batch = Batch(
        a=np.float64(1.0),
        b=Batch(c=np.ones((1,), dtype=np.float32), d=torch.ones((1,), dtype=torch.float64)),
    )
    a_torch_float = to_torch(batch.a, dtype=torch.float32)
    assert a_torch_float.dtype == torch.float32
    a_torch_double = to_torch(batch.a, dtype=torch.float64)
    assert a_torch_double.dtype == torch.float64
    batch_torch_float = to_torch(batch, dtype=torch.float32)
    assert batch_torch_float.a.dtype == torch.float32
    assert batch_torch_float.b.c.dtype == torch.float32
    assert batch_torch_float.b.d.dtype == torch.float32
    data_list = [float("nan"), 1]
    data_list_torch = to_torch(data_list)
    assert data_list_torch.dtype == torch.float64
    data_list_2 = [np.random.rand(3, 3), np.random.rand(3, 3)]
    data_list_2_torch = to_torch(data_list_2)
    assert data_list_2_torch.shape == (2, 3, 3)
    assert np.allclose(to_numpy(to_torch(data_list_2)), data_list_2)
    data_list_3 = [np.zeros((3, 2)), np.zeros((3, 3))]
    data_list_3_torch = [torch.zeros((3, 2)), torch.zeros((3, 3))]
    with pytest.raises(TypeError):
        to_torch(data_list_3)
    with pytest.raises(TypeError):
        to_numpy(data_list_3_torch)
    data_list_4 = [np.zeros((2, 3)), np.zeros((3, 3))]
    data_list_4_torch = [torch.zeros((2, 3)), torch.zeros((3, 3))]
    with pytest.raises(TypeError):
        to_torch(data_list_4)
    with pytest.raises(TypeError):
        to_numpy(data_list_4_torch)
    data_list_5 = [np.zeros(2), np.zeros((3, 3))]
    data_list_5_torch = [torch.zeros(2), torch.zeros((3, 3))]
    with pytest.raises(TypeError):
        to_torch(data_list_5)
    with pytest.raises(TypeError):
        to_numpy(data_list_5_torch)
    data_array = np.random.rand(3, 2, 2)
    data_empty_tensor = to_torch(data_array[[]])
    assert isinstance(data_empty_tensor, torch.Tensor)
    assert data_empty_tensor.shape == (0, 2, 2)
    data_empty_array = to_numpy(data_empty_tensor)
    assert isinstance(data_empty_array, np.ndarray)
    assert data_empty_array.shape == (0, 2, 2)
    assert np.allclose(to_numpy(to_torch(data_array)), data_array)
    # additional test for to_numpy, for code-coverage
    assert isinstance(to_numpy(1), np.ndarray)
    assert isinstance(to_numpy(1.0), np.ndarray)
    assert isinstance(to_numpy({"a": torch.tensor(1)})["a"], np.ndarray)
    assert isinstance(to_numpy(Batch(a=torch.tensor(1))).a, np.ndarray)
    assert to_numpy(None).item() is None
    assert to_numpy(to_numpy).item() == to_numpy
    # additional test for to_torch, for code-coverage
    assert isinstance(to_torch(1), torch.Tensor)
    if sys.platform in ["win32", "cygwin"]:  # windows
        assert to_torch(1).dtype == torch.int32
    else:
        assert to_torch(1).dtype == torch.int64
    assert to_torch(1.0).dtype == torch.float64
    assert isinstance(to_torch({"a": [1]})["a"], torch.Tensor)
    with pytest.raises(TypeError):
        to_torch(None)
    with pytest.raises(TypeError):
        to_torch(np.array([{}, "2"]))


def test_batch_pickle() -> None:
    batch = Batch(obs=Batch(a=0.0, c=torch.Tensor([1.0, 2.0])), np=np.zeros([3, 4]))
    batch_pk = pickle.loads(pickle.dumps(batch))
    assert batch.obs.a == batch_pk.obs.a
    assert torch.all(batch.obs.c == batch_pk.obs.c)
    assert np.all(batch.np == batch_pk.np)


def test_batch_from_to_numpy_without_copy() -> None:
    batch = Batch(a=np.ones((1,)), b=Batch(c=np.ones((1,))))
    a_mem_addr_orig = batch.a.__array_interface__["data"][0]
    c_mem_addr_orig = batch.b.c.__array_interface__["data"][0]
    batch.to_torch_()
    batch.to_numpy_()
    a_mem_addr_new = batch.a.__array_interface__["data"][0]
    c_mem_addr_new = batch.b.c.__array_interface__["data"][0]
    assert a_mem_addr_new == a_mem_addr_orig
    assert c_mem_addr_new == c_mem_addr_orig


def test_batch_copy() -> None:
    batch = Batch(a=np.array([3, 4, 5]), b=np.array([4, 5, 6]))
    batch2 = Batch({"c": np.array([6, 7, 8]), "b": batch})
    orig_c_addr = batch2.c.__array_interface__["data"][0]
    orig_b_a_addr = batch2.b.a.__array_interface__["data"][0]
    orig_b_b_addr = batch2.b.b.__array_interface__["data"][0]
    # test with copy=False
    batch3 = Batch(copy=False, **batch2)
    curr_c_addr = batch3.c.__array_interface__["data"][0]
    curr_b_a_addr = batch3.b.a.__array_interface__["data"][0]
    curr_b_b_addr = batch3.b.b.__array_interface__["data"][0]
    assert batch2.c is batch3.c
    assert batch2.b is batch3.b
    assert batch2.b.a is batch3.b.a
    assert batch2.b.b is batch3.b.b
    assert orig_c_addr == curr_c_addr
    assert orig_b_a_addr == curr_b_a_addr
    assert orig_b_b_addr == curr_b_b_addr
    # test with copy=True
    batch3 = Batch(copy=True, **batch2)
    curr_c_addr = batch3.c.__array_interface__["data"][0]
    curr_b_a_addr = batch3.b.a.__array_interface__["data"][0]
    curr_b_b_addr = batch3.b.b.__array_interface__["data"][0]
    assert batch2.c is not batch3.c
    assert batch2.b is not batch3.b
    assert batch2.b.a is not batch3.b.a
    assert batch2.b.b is not batch3.b.b
    assert orig_c_addr != curr_c_addr
    assert orig_b_a_addr != curr_b_a_addr
    assert orig_b_b_addr != curr_b_b_addr


def test_batch_empty() -> None:
    b5_dict = np.array([{"a": False, "b": {"c": 2.0, "d": 1.0}}, {"a": True, "b": {"c": 3.0}}])
    b5 = Batch(b5_dict)
    b5[1] = Batch.empty(b5[0])
    assert np.allclose(b5.a, [False, False])
    assert np.allclose(b5.b.c, [2, 0])
    assert np.allclose(b5.b.d, [1, 0])
    data = Batch(
        a=[False, True],
        b={
            "c": np.array([2.0, "st"], dtype=object),
            "d": [1, None],
            "e": [2.0, float("nan")],
        },
        c=np.array([1, 3, 4], dtype=int),
        t=torch.tensor([4, 5, 6, 7.0]),
    )
    data[-1] = Batch.empty(data[1])
    assert np.allclose(data.c, [1, 3, 0])
    assert np.allclose(data.a, [False, False])
    assert list(data.b.c) == [2.0, None]
    assert list(data.b.d) == [1, None]
    assert np.allclose(data.b.e, [2, 0])
    assert torch.allclose(data.t, torch.tensor([4, 5, 6, 0.0]))
    data[0].empty_()  # which will fail in a, b.c, b.d, b.e, c
    assert torch.allclose(data.t, torch.tensor([0.0, 5, 6, 0]))
    data.empty_(index=0)
    assert np.allclose(data.c, [0, 3, 0])
    assert list(data.b.c) == [None, None]
    assert list(data.b.d) == [None, None]
    assert list(data.b.e) == [0, 0]
    b0 = Batch()
    b0.empty_()
    assert b0.shape == []


def test_batch_standard_compatibility() -> None:
    batch = Batch(a=np.array([[1.0, 2.0], [3.0, 4.0]]), b=Batch(), c=np.array([5.0, 6.0]))
    batch_mean = np.mean(batch)
    assert isinstance(batch_mean, Batch)  # type: ignore  # mypy doesn't know but it works, cf. `batch.rst`
    assert sorted(batch_mean.get_keys()) == ["a", "b", "c"]  # type: ignore
    with pytest.raises(TypeError):
        len(batch_mean)
    assert np.all(batch_mean.a == np.mean(batch.a, axis=0))
    assert batch_mean.c == np.mean(batch.c, axis=0)
    with pytest.raises(IndexError):
        Batch()[0]


class TestBatchEquality:
    @staticmethod
    def test_keys_different() -> None:
        batch1 = Batch(a=[1, 2], b=[100, 50])
        batch2 = Batch(b=[1, 2], c=[100, 50])
        assert batch1 != batch2

    @staticmethod
    def test_keys_missing() -> None:
        batch1 = Batch(a=[1, 2], b=[2, 3, 4])
        batch2 = Batch(a=[1, 2], b=[2, 3, 4])
        batch2.pop("b")
        assert batch1 != batch2

    @staticmethod
    def test_types_keys_different() -> None:
        batch1 = Batch(a=[1, 2, 3], b=[4, 5])
        batch2 = Batch(a=[1, 2, 3], b=Batch(a=[4, 5]))
        assert batch1 != batch2

    @staticmethod
    def test_array_types_different() -> None:
        batch1 = Batch(a=[1, 2, 3], b=np.array([4, 5]))
        batch2 = Batch(a=[1, 2, 3], b=torch.Tensor([4, 5]))
        assert batch1 != batch2

    @staticmethod
    def test_nested_values_different() -> None:
        batch1 = Batch(a=Batch(a=[1, 2, 3]), b=[4, 5])
        batch2 = Batch(a=Batch(a=[1, 2, 4]), b=[4, 5])
        assert batch1 != batch2

    @staticmethod
    def test_nested_shapes_different() -> None:
        batch1 = Batch(a=Batch(a=[1, 2, 3]), b=[4, 5])
        batch2 = Batch(a=Batch(a=[1, 4]), b=[4, 5])
        assert batch1 != batch2

    @staticmethod
    def test_slice_equal() -> None:
        batch1 = Batch(a=[1, 2, 3])
        assert batch1[:2] == batch1[:2]

    @staticmethod
    def test_slice_ellipsis_equal() -> None:
        batch1 = Batch(a=Batch(a=[1, 2, 3]), b=[4, 5], c=[100, 1001, 2000])
        assert batch1[..., 1:] == batch1[..., 1:]

    @staticmethod
    def test_empty_batches() -> None:
        assert Batch() == Batch()

    @staticmethod
    def test_different_order_keys() -> None:
        assert Batch(a=1, b=2) == Batch(b=2, a=1)

    @staticmethod
    def test_tuple_and_list_types() -> None:
        assert Batch(a=(1, 2)) == Batch(a=[1, 2])

    @staticmethod
    def test_subbatch_dict_and_batch_types() -> None:
        assert Batch(a={"x": 1}) == Batch(a=Batch(x=1))


class TestBatchToDict:
    @staticmethod
    def test_to_dict_empty_batch_no_recurse() -> None:
        batch = Batch()
        expected: dict[Any, Any] = {}
        assert batch.to_dict() == expected

    @staticmethod
    def test_to_dict_with_simple_values_recurse() -> None:
        batch = Batch(a=1, b="two", c=np.array([3, 4]))
        expected = {"a": np.asanyarray(1), "b": "two", "c": np.array([3, 4])}
        assert not DeepDiff(batch.to_dict(recursive=True), expected)

    @staticmethod
    def test_to_dict_simple() -> None:
        batch = Batch(a=1, b="two")
        expected = {"a": np.asanyarray(1), "b": "two"}
        assert batch.to_dict() == expected

    @staticmethod
    def test_to_dict_nested_batch_no_recurse() -> None:
        nested_batch = Batch(c=3)
        batch = Batch(a=1, b=nested_batch)
        expected = {"a": np.asanyarray(1), "b": nested_batch}
        assert not DeepDiff(batch.to_dict(recursive=False), expected)

    @staticmethod
    def test_to_dict_nested_batch_recurse() -> None:
        nested_batch = Batch(c=3)
        batch = Batch(a=1, b=nested_batch)
        expected = {"a": np.asanyarray(1), "b": {"c": np.asanyarray(3)}}
        assert not DeepDiff(batch.to_dict(recursive=True), expected)

    @staticmethod
    def test_to_dict_multiple_nested_batch_recurse() -> None:
        nested_batch = Batch(c=Batch(e=3), d=[100, 200, 300])
        batch = Batch(a=1, b=nested_batch)
        expected = {
            "a": np.asanyarray(1),
            "b": {"c": {"e": np.asanyarray(3)}, "d": np.array([100, 200, 300])},
        }
        assert not DeepDiff(batch.to_dict(recursive=True), expected)

    @staticmethod
    def test_to_dict_array() -> None:
        batch = Batch(a=np.array([1, 2, 3]))
        expected = {"a": np.array([1, 2, 3])}
        assert not DeepDiff(batch.to_dict(), expected)

    @staticmethod
    def test_to_dict_nested_batch_with_array() -> None:
        nested_batch = Batch(c=np.array([4, 5]))
        batch = Batch(a=1, b=nested_batch)
        expected = {"a": np.asanyarray(1), "b": {"c": np.array([4, 5])}}
        assert not DeepDiff(batch.to_dict(recursive=True), expected)

    @staticmethod
    def test_to_dict_torch_tensor() -> None:
        t1 = torch.tensor([1.0, 2.0]).detach().cpu().numpy()
        batch = Batch(a=t1)
        t2 = torch.tensor([1.0, 2.0]).detach().cpu().numpy()
        expected = {"a": t2}
        assert not DeepDiff(batch.to_dict(), expected)

    @staticmethod
    def test_to_dict_nested_batch_with_torch_tensor() -> None:
        nested_batch = Batch(c=torch.tensor([4, 5]).detach().cpu().numpy())
        batch = Batch(a=1, b=nested_batch)
        expected = {"a": np.asanyarray(1), "b": {"c": torch.tensor([4, 5]).detach().cpu().numpy()}}
        assert not DeepDiff(batch.to_dict(recursive=True), expected)


class TestToNumpy:
    """Tests for `Batch.to_numpy()` and its in-place counterpart `Batch.to_numpy_()` ."""

    @staticmethod
    def test_to_numpy() -> None:
        batch = Batch(a=1, b=torch.arange(5), c={"d": torch.tensor([1, 2, 3])})
        new_batch: Batch = Batch.to_numpy(batch)
        assert id(batch) != id(new_batch)
        assert isinstance(batch.b, torch.Tensor)
        assert isinstance(batch.c.d, torch.Tensor)

        assert isinstance(new_batch.b, np.ndarray)
        assert isinstance(new_batch.c.d, np.ndarray)

    @staticmethod
    def test_to_numpy_() -> None:
        batch = Batch(a=1, b=torch.arange(5), c={"d": torch.tensor([1, 2, 3])})
        id_batch = id(batch)
        batch.to_numpy_()
        assert id_batch == id(batch)
        assert isinstance(batch.b, np.ndarray)
        assert isinstance(batch.c.d, np.ndarray)


class TestToTorch:
    """Tests for `Batch.to_torch()` and its in-place counterpart `Batch.to_torch_()` ."""

    @staticmethod
    def test_to_torch() -> None:
        batch = Batch(a=1, b=np.arange(5), c={"d": np.array([1, 2, 3])})
        new_batch: Batch = Batch.to_torch(batch)
        assert id(batch) != id(new_batch)
        assert isinstance(batch.b, np.ndarray)
        assert isinstance(batch.c.d, np.ndarray)

        assert isinstance(new_batch.b, torch.Tensor)
        assert isinstance(new_batch.c.d, torch.Tensor)

    @staticmethod
    def test_to_torch_() -> None:
        batch = Batch(a=1, b=np.arange(5), c={"d": np.array([1, 2, 3])})
        id_batch = id(batch)
        batch.to_torch_()
        assert id_batch == id(batch)
        assert isinstance(batch.b, torch.Tensor)
        assert isinstance(batch.c.d, torch.Tensor)
