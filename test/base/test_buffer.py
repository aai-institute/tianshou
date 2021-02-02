import os
import torch
import pickle
import pytest
import tempfile
import h5py
import numpy as np
from timeit import timeit

from tianshou.data.utils.converter import to_hdf5
from tianshou.data import Batch, SegmentTree, ReplayBuffer
from tianshou.data import ListReplayBuffer, PrioritizedReplayBuffer
from tianshou.data import ReplayBufferManager, CachedReplayBuffer

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


def test_replaybuffer(size=10, bufsize=20):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize)
    buf.update(buf)
    assert str(buf) == buf.__class__.__name__ + '()'
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, [a], rew, done, obs_next, info)
        obs = obs_next
        assert len(buf) == min(bufsize, i + 1)
    with pytest.raises(ValueError):
        buf._add_to_buffer('rew', np.array([1, 2, 3]))
    assert buf.act.dtype == np.object
    assert isinstance(buf.act[0], list)
    data, indice = buf.sample(bufsize * 2)
    assert (indice < len(buf)).all()
    assert (data.obs < size).all()
    assert (0 <= data.done).all() and (data.done <= 1).all()
    b = ReplayBuffer(size=10)
    # neg bsz should return empty index
    assert b.sample_index(-1).tolist() == []
    b.add(1, 1, 1, 1, 'str', {'a': 3, 'b': {'c': 5.0}})
    assert b.obs[0] == 1
    assert b.done[0]
    assert b.obs_next[0] == 'str'
    assert np.all(b.obs[1:] == 0)
    assert np.all(b.obs_next[1:] == np.array(None))
    assert b.info.a[0] == 3 and b.info.a.dtype == np.integer
    assert np.all(b.info.a[1:] == 0)
    assert b.info.b.c[0] == 5.0 and b.info.b.c.dtype == np.inexact
    assert np.all(b.info.b.c[1:] == 0.0)
    with pytest.raises(IndexError):
        b[22]
    b = ListReplayBuffer()
    with pytest.raises(NotImplementedError):
        b.sample(0)


def test_ignore_obs_next(size=10):
    # Issue 82
    buf = ReplayBuffer(size, ignore_obs_next=True)
    for i in range(size):
        buf.add(obs={'mask1': np.array([i, 1, 1, 0, 0]),
                     'mask2': np.array([i + 4, 0, 1, 0, 0]),
                     'mask': i},
                act={'act_id': i,
                     'position_id': i + 3},
                rew=i,
                done=i % 3 == 0,
                info={'if': i})
    indice = np.arange(len(buf))
    orig = np.arange(len(buf))
    data = buf[indice]
    data2 = buf[indice]
    assert isinstance(data, Batch)
    assert isinstance(data2, Batch)
    assert np.allclose(indice, orig)
    assert np.allclose(data.obs_next.mask, data2.obs_next.mask)
    assert np.allclose(data.obs_next.mask, [0, 2, 3, 3, 5, 6, 6, 8, 9, 9])
    buf.stack_num = 4
    data = buf[indice]
    data2 = buf[indice]
    assert np.allclose(data.obs_next.mask, data2.obs_next.mask)
    assert np.allclose(data.obs_next.mask, np.array([
        [0, 0, 0, 0], [1, 1, 1, 2], [1, 1, 2, 3], [1, 1, 2, 3],
        [4, 4, 4, 5], [4, 4, 5, 6], [4, 4, 5, 6],
        [7, 7, 7, 8], [7, 7, 8, 9], [7, 7, 8, 9]]))
    assert np.allclose(data.info['if'], data2.info['if'])
    assert np.allclose(data.info['if'], np.array([
        [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3],
        [4, 4, 4, 4], [4, 4, 4, 5], [4, 4, 5, 6],
        [7, 7, 7, 7], [7, 7, 7, 8], [7, 7, 8, 9]]))
    assert data.obs_next


def test_stack(size=5, bufsize=9, stack_num=4, cached_num=3):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize, stack_num=stack_num)
    buf2 = ReplayBuffer(bufsize, stack_num=stack_num, sample_avail=True)
    buf3 = ReplayBuffer(bufsize, stack_num=stack_num, save_only_last_obs=True)
    obs = env.reset(1)
    for i in range(16):
        obs_next, rew, done, info = env.step(1)
        buf.add(obs, 1, rew, done, None, info)
        buf2.add(obs, 1, rew, done, None, info)
        buf3.add([None, None, obs], 1, rew, done, [None, obs], info)
        obs = obs_next
        if done:
            obs = env.reset(1)
    indice = np.arange(len(buf))
    assert np.allclose(buf.get(indice, 'obs')[..., 0], [
        [1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 4],
        [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3],
        [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 1, 1]])
    assert np.allclose(buf.get(indice, 'obs'), buf3.get(indice, 'obs'))
    assert np.allclose(buf.get(indice, 'obs'), buf3.get(indice, 'obs_next'))
    _, indice = buf2.sample(0)
    assert indice.tolist() == [2, 6]
    _, indice = buf2.sample(1)
    assert indice[0] in [2, 6]
    batch, indice = buf2.sample(-1)  # neg bsz -> no data
    assert indice.tolist() == [] and len(batch) == 0
    with pytest.raises(IndexError):
        buf[bufsize * 2]


def test_priortized_replaybuffer(size=32, bufsize=15):
    env = MyTestEnv(size)
    buf = PrioritizedReplayBuffer(bufsize, 0.5, 0.5)
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, a, rew, done, obs_next, info, np.random.randn() - 0.5)
        obs = obs_next
        data, indice = buf.sample(len(buf) // 2)
        if len(buf) // 2 == 0:
            assert len(data) == len(buf)
        else:
            assert len(data) == len(buf) // 2
        assert len(buf) == min(bufsize, i + 1)
    data, indice = buf.sample(len(buf) // 2)
    buf.update_weight(indice, -data.weight / 2)
    assert np.allclose(
        buf.weight[indice], np.abs(-data.weight / 2) ** buf._alpha)


def test_update():
    buf1 = ReplayBuffer(4, stack_num=2)
    buf2 = ReplayBuffer(4, stack_num=2)
    for i in range(5):
        buf1.add(obs=np.array([i]), act=float(i), rew=i * i,
                 done=i % 2 == 0, info={'incident': 'found'})
    assert len(buf1) > len(buf2)
    buf2.update(buf1)
    assert len(buf1) == len(buf2)
    assert (buf2[0].obs == buf1[1].obs).all()
    assert (buf2[-1].obs == buf1[0].obs).all()
    b = ListReplayBuffer()
    with pytest.raises(NotImplementedError):
        b.update(b)
    b = CachedReplayBuffer(ReplayBuffer(10), 4, 5)
    with pytest.raises(NotImplementedError):
        b.update(b)


def test_segtree():
    realop = np.sum
    # small test
    actual_len = 8
    tree = SegmentTree(actual_len)  # 1-15. 8-15 are leaf nodes
    assert len(tree) == actual_len
    assert np.all([tree[i] == 0. for i in range(actual_len)])
    with pytest.raises(IndexError):
        tree[actual_len]
    naive = np.zeros([actual_len])
    for _ in range(1000):
        # random choose a place to perform single update
        index = np.random.randint(actual_len)
        value = np.random.rand()
        naive[index] = value
        tree[index] = value
        for i in range(actual_len):
            for j in range(i + 1, actual_len):
                ref = realop(naive[i:j])
                out = tree.reduce(i, j)
                assert np.allclose(ref, out), (ref, out)
    assert np.allclose(tree.reduce(start=1), realop(naive[1:]))
    assert np.allclose(tree.reduce(end=-1), realop(naive[:-1]))
    # batch setitem
    for _ in range(1000):
        index = np.random.choice(actual_len, size=4)
        value = np.random.rand(4)
        naive[index] = value
        tree[index] = value
        assert np.allclose(realop(naive), tree.reduce())
        for i in range(10):
            left = np.random.randint(actual_len)
            right = np.random.randint(left + 1, actual_len + 1)
            assert np.allclose(realop(naive[left:right]),
                               tree.reduce(left, right))
    # large test
    actual_len = 16384
    tree = SegmentTree(actual_len)
    naive = np.zeros([actual_len])
    for _ in range(1000):
        index = np.random.choice(actual_len, size=64)
        value = np.random.rand(64)
        naive[index] = value
        tree[index] = value
        assert np.allclose(realop(naive), tree.reduce())
        for i in range(10):
            left = np.random.randint(actual_len)
            right = np.random.randint(left + 1, actual_len + 1)
            assert np.allclose(realop(naive[left:right]),
                               tree.reduce(left, right))

    # test prefix-sum-idx
    actual_len = 8
    tree = SegmentTree(actual_len)
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar <= naive[:index + 1].sum()
    # corner case here
    naive = np.ones(actual_len, np.int)
    tree[np.arange(actual_len)] = naive
    for scalar in range(actual_len):
        index = tree.get_prefix_sum_idx(scalar * 1.)
        assert naive[:index].sum() <= scalar <= naive[:index + 1].sum()
    tree = SegmentTree(10)
    tree[np.arange(3)] = np.array([0.1, 0, 0.1])
    assert np.allclose(tree.get_prefix_sum_idx(
        np.array([0, .1, .1 + 1e-6, .2 - 1e-6])), [0, 0, 2, 2])
    with pytest.raises(AssertionError):
        tree.get_prefix_sum_idx(.2)
    # test large prefix-sum-idx
    actual_len = 16384
    tree = SegmentTree(actual_len)
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar <= naive[:index + 1].sum()

    # profile
    if __name__ == '__main__':
        size = 100000
        bsz = 64
        naive = np.random.rand(size)
        tree = SegmentTree(size)
        tree[np.arange(size)] = naive

        def sample_npbuf():
            return np.random.choice(size, bsz, p=naive / naive.sum())

        def sample_tree():
            scalar = np.random.rand(bsz) * tree.reduce()
            return tree.get_prefix_sum_idx(scalar)

        print('npbuf', timeit(sample_npbuf, setup=sample_npbuf, number=1000))
        print('tree', timeit(sample_tree, setup=sample_tree, number=1000))


def test_pickle():
    size = 100
    vbuf = ReplayBuffer(size, stack_num=2)
    lbuf = ListReplayBuffer()
    pbuf = PrioritizedReplayBuffer(size, 0.6, 0.4)
    rew = np.array([1, 1])
    for i in range(4):
        vbuf.add(obs=Batch(index=np.array([i])), act=0, rew=rew, done=0)
    for i in range(3):
        lbuf.add(obs=Batch(index=np.array([i])), act=1, rew=rew, done=0)
    for i in range(5):
        pbuf.add(obs=Batch(index=np.array([i])),
                 act=2, rew=rew, done=0, weight=np.random.rand())
    # save & load
    _vbuf = pickle.loads(pickle.dumps(vbuf))
    _lbuf = pickle.loads(pickle.dumps(lbuf))
    _pbuf = pickle.loads(pickle.dumps(pbuf))
    assert len(_vbuf) == len(vbuf) and np.allclose(_vbuf.act, vbuf.act)
    assert len(_lbuf) == len(lbuf) and np.allclose(_lbuf.act, lbuf.act)
    assert len(_pbuf) == len(pbuf) and np.allclose(_pbuf.act, pbuf.act)
    # make sure the meta var is identical
    assert _vbuf.stack_num == vbuf.stack_num
    assert np.allclose(_pbuf.weight[np.arange(len(_pbuf))],
                       pbuf.weight[np.arange(len(pbuf))])


def test_hdf5():
    size = 100
    buffers = {
        "array": ReplayBuffer(size, stack_num=2),
        "list": ListReplayBuffer(),
        "prioritized": PrioritizedReplayBuffer(size, 0.6, 0.4),
    }
    buffer_types = {k: b.__class__ for k, b in buffers.items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    info_t = torch.tensor([1.]).to(device)
    for i in range(4):
        kwargs = {
            'obs': Batch(index=np.array([i])),
            'act': i,
            'rew': np.array([1, 2]),
            'done': i % 3 == 2,
            'info': {"number": {"n": i, "t": info_t}, 'extra': None},
        }
        buffers["array"].add(**kwargs)
        buffers["list"].add(**kwargs)
        buffers["prioritized"].add(weight=np.random.rand(), **kwargs)

    # save
    paths = {}
    for k, buf in buffers.items():
        f, path = tempfile.mkstemp(suffix='.hdf5')
        os.close(f)
        buf.save_hdf5(path)
        paths[k] = path

    # load replay buffer
    _buffers = {k: buffer_types[k].load_hdf5(paths[k]) for k in paths.keys()}

    # compare
    for k in buffers.keys():
        assert len(_buffers[k]) == len(buffers[k])
        assert np.allclose(_buffers[k].act, buffers[k].act)
        assert _buffers[k].stack_num == buffers[k].stack_num
        assert _buffers[k].maxsize == buffers[k].maxsize
        assert np.all(_buffers[k]._indices == buffers[k]._indices)
    for k in ["array", "prioritized"]:
        assert _buffers[k]._index == buffers[k]._index
        assert isinstance(buffers[k].get(0, "info"), Batch)
        assert isinstance(_buffers[k].get(0, "info"), Batch)
    for k in ["array"]:
        assert np.all(
            buffers[k][:].info.number.n == _buffers[k][:].info.number.n)
        assert np.all(
            buffers[k][:].info.extra == _buffers[k][:].info.extra)

    # raise exception when value cannot be pickled
    data = {"not_supported": lambda x: x * x}
    grp = h5py.Group
    with pytest.raises(NotImplementedError):
        to_hdf5(data, grp)
    # ndarray with data type not supported by HDF5 that cannot be pickled
    data = {"not_supported": np.array(lambda x: x * x)}
    grp = h5py.Group
    with pytest.raises(RuntimeError):
        to_hdf5(data, grp)


def test_replaybuffermanager():
    buf = ReplayBufferManager([ReplayBuffer(size=5) for i in range(4)])
    ep_len, ep_rew = buf.add(obs=[1, 2, 3], act=[1, 2, 3], rew=[1, 2, 3],
                             done=[0, 0, 1], buffer_ids=[0, 1, 2])
    assert np.allclose(ep_len, [0, 0, 1]) and np.allclose(ep_rew, [0, 0, 3])
    with pytest.raises(NotImplementedError):
        # ReplayBufferManager cannot be updated
        buf.update(buf)
    # sample index / prev / next / unfinished_index
    indice = buf.sample_index(11000)
    assert np.bincount(indice)[[0, 5, 10]].min() >= 3000  # uniform sample
    batch, indice = buf.sample(0)
    assert np.allclose(indice, [0, 5, 10])
    indice_prev = buf.prev(indice)
    assert np.allclose(indice_prev, indice), indice_prev
    indice_next = buf.next(indice)
    assert np.allclose(indice_next, indice), indice_next
    assert np.allclose(buf.unfinished_index(), [0, 5])
    buf.add(obs=[4], act=[4], rew=[4], done=[1], buffer_ids=[3])
    assert np.allclose(buf.unfinished_index(), [0, 5])
    batch, indice = buf.sample(10)
    batch, indice = buf.sample(0)
    assert np.allclose(indice, [0, 5, 10, 15])
    indice_prev = buf.prev(indice)
    assert np.allclose(indice_prev, indice), indice_prev
    indice_next = buf.next(indice)
    assert np.allclose(indice_next, indice), indice_next
    data = np.array([0, 0, 0, 0])
    buf.add(obs=data, act=data, rew=data, done=data, buffer_ids=[0, 1, 2, 3])
    buf.add(obs=data, act=data, rew=data, done=1 - data,
            buffer_ids=[0, 1, 2, 3])
    assert len(buf) == 12
    buf.add(obs=data, act=data, rew=data, done=data, buffer_ids=[0, 1, 2, 3])
    buf.add(obs=data, act=data, rew=data, done=[0, 1, 0, 1],
            buffer_ids=[0, 1, 2, 3])
    assert len(buf) == 20
    indice = buf.sample_index(120000)
    assert np.bincount(indice).min() >= 5000
    batch, indice = buf.sample(10)
    indice = buf.sample_index(0)
    assert np.allclose(indice, np.arange(len(buf)))
    # check the actual data stored in buf._meta
    assert np.allclose(buf.done, [
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 1,
        1, 0, 1, 0, 0,
        1, 0, 1, 0, 1,
    ])
    assert np.allclose(buf.prev(indice), [
        0, 0, 1, 3, 3,
        5, 5, 6, 8, 8,
        10, 11, 11, 13, 13,
        15, 16, 16, 18, 18,
    ])
    assert np.allclose(buf.next(indice), [
        1, 2, 2, 4, 4,
        6, 7, 7, 9, 9,
        10, 12, 12, 14, 14,
        15, 17, 17, 19, 19,
    ])
    assert np.allclose(buf.unfinished_index(), [4, 14])
    ep_len, ep_rew = buf.add(obs=[1], act=[1], rew=[1], done=[1],
                             buffer_ids=[2])
    assert np.allclose(ep_len, [3]) and np.allclose(ep_rew, [1])
    assert np.allclose(buf.unfinished_index(), [4])
    indice = list(sorted(buf.sample_index(0)))
    assert np.allclose(indice, np.arange(len(buf)))
    assert np.allclose(buf.prev(indice), [
        0, 0, 1, 3, 3,
        5, 5, 6, 8, 8,
        14, 11, 11, 13, 13,
        15, 16, 16, 18, 18,
    ])
    assert np.allclose(buf.next(indice), [
        1, 2, 2, 4, 4,
        6, 7, 7, 9, 9,
        10, 12, 12, 14, 10,
        15, 17, 17, 19, 19,
    ])
    # corner case: list, int and -1
    assert buf.prev(-1) == buf.prev([buf.maxsize - 1])[0]
    assert buf.next(-1) == buf.next([buf.maxsize - 1])[0]
    batch = buf._meta
    batch.info.n = np.ones(buf.maxsize)
    buf.set_batch(batch)
    assert np.allclose(buf.buffers[-1].info.n, [1] * 5)
    assert buf.sample_index(-1).tolist() == []
    assert np.array([ReplayBuffer(0, ignore_obs_next=True)]).dtype == np.object


def test_cachedbuffer():
    buf = CachedReplayBuffer(ReplayBuffer(10), 4, 5)
    assert buf.sample_index(0).tolist() == []
    # check the normal function/usage/storage in CachedReplayBuffer
    ep_len, ep_rew = buf.add(obs=[1], act=[1], rew=[1], done=[0],
                             cached_buffer_ids=[1])
    obs = np.zeros(buf.maxsize)
    obs[15] = 1
    indice = buf.sample_index(0)
    assert np.allclose(indice, [15])
    assert np.allclose(buf.prev(indice), [15])
    assert np.allclose(buf.next(indice), [15])
    assert np.allclose(buf.obs, obs)
    assert np.allclose(ep_len, [0]) and np.allclose(ep_rew, [0.0])
    ep_len, ep_rew = buf.add(obs=[2], act=[2], rew=[2], done=[1],
                             cached_buffer_ids=[3])
    obs[[0, 25]] = 2
    indice = buf.sample_index(0)
    assert np.allclose(indice, [0, 15])
    assert np.allclose(buf.prev(indice), [0, 15])
    assert np.allclose(buf.next(indice), [0, 15])
    assert np.allclose(buf.obs, obs)
    assert np.allclose(ep_len, [1]) and np.allclose(ep_rew, [2.0])
    assert np.allclose(buf.unfinished_index(), [15])
    assert np.allclose(buf.sample_index(0), [0, 15])
    ep_len, ep_rew = buf.add(obs=[3, 4], act=[3, 4], rew=[3, 4],
                             done=[0, 1], cached_buffer_ids=[3, 1])
    assert np.allclose(ep_len, [0, 2]) and np.allclose(ep_rew, [0, 5.0])
    obs[[0, 1, 2, 15, 16, 25]] = [2, 1, 4, 1, 4, 3]
    assert np.allclose(buf.obs, obs)
    assert np.allclose(buf.unfinished_index(), [25])
    indice = buf.sample_index(0)
    assert np.allclose(indice, [0, 1, 2, 25])
    assert np.allclose(buf.done[indice], [1, 0, 1, 0])
    assert np.allclose(buf.prev(indice), [0, 1, 1, 25])
    assert np.allclose(buf.next(indice), [0, 2, 2, 25])
    indice = buf.sample_index(10000)
    assert np.bincount(indice)[[0, 1, 2, 25]].min() > 2000  # uniform sample
    # cached buffer with main_buffer size == 0 (no update)
    # used in test_collector
    buf = CachedReplayBuffer(ReplayBuffer(0, sample_avail=True), 4, 5)
    data = np.zeros(4)
    rew = np.ones([4, 4])
    buf.add(obs=data, act=data, rew=rew, done=[0, 0, 1, 1], obs_next=data)
    buf.add(obs=data, act=data, rew=rew, done=[0, 0, 0, 0], obs_next=data)
    buf.add(obs=data, act=data, rew=rew, done=[1, 1, 1, 1], obs_next=data)
    buf.add(obs=data, act=data, rew=rew, done=[0, 0, 0, 0], obs_next=data)
    buf.add(obs=data, act=data, rew=rew, done=[0, 1, 0, 1], obs_next=data)
    assert np.allclose(buf.done, [
        0, 0, 1, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
    ])
    indice = buf.sample_index(0)
    assert np.allclose(indice, [0, 1, 10, 11])
    assert np.allclose(buf.prev(indice), [0, 0, 10, 10])
    assert np.allclose(buf.next(indice), [1, 1, 11, 11])


def test_multibuf_stack():
    size = 5
    bufsize = 9
    stack_num = 4
    cached_num = 3
    env = MyTestEnv(size)
    # test if CachedReplayBuffer can handle stack_num + ignore_obs_next
    buf4 = CachedReplayBuffer(
        ReplayBuffer(bufsize, stack_num=stack_num, ignore_obs_next=True),
        cached_num, size)
    # test if CachedReplayBuffer can handle super corner case:
    # prio-buffer + stack_num + ignore_obs_next + sample_avail
    buf5 = CachedReplayBuffer(
        PrioritizedReplayBuffer(bufsize, 0.6, 0.4, stack_num=stack_num,
                                ignore_obs_next=True, sample_avail=True),
        cached_num, size)
    obs = env.reset(1)
    for i in range(18):
        obs_next, rew, done, info = env.step(1)
        obs_list = np.array([obs + size * i for i in range(cached_num)])
        act_list = [1] * cached_num
        rew_list = [rew] * cached_num
        done_list = [done] * cached_num
        obs_next_list = -obs_list
        info_list = [info] * cached_num
        buf4.add(obs_list, act_list, rew_list, done_list,
                 obs_next_list, info_list)
        buf5.add(obs_list, act_list, rew_list, done_list,
                 obs_next_list, info_list)
        obs = obs_next
        if done:
            obs = env.reset(1)
    # check the `add` order is correct
    assert np.allclose(buf4.obs.reshape(-1), [
        12, 13, 14, 4, 6, 7, 8, 9, 11,  # main_buffer
        1, 2, 3, 4, 0,  # cached_buffer[0]
        6, 7, 8, 9, 0,  # cached_buffer[1]
        11, 12, 13, 14, 0,  # cached_buffer[2]
    ]), buf4.obs
    assert np.allclose(buf4.done, [
        0, 0, 1, 1, 0, 0, 0, 1, 0,  # main_buffer
        0, 0, 0, 1, 0,  # cached_buffer[0]
        0, 0, 0, 1, 0,  # cached_buffer[1]
        0, 0, 0, 1, 0,  # cached_buffer[2]
    ]), buf4.done
    assert np.allclose(buf4.unfinished_index(), [10, 15, 20])
    indice = sorted(buf4.sample_index(0))
    assert np.allclose(indice, list(range(bufsize)) + [9, 10, 14, 15, 19, 20])
    assert np.allclose(buf4[indice].obs[..., 0], [
        [11, 11, 11, 12], [11, 11, 12, 13], [11, 12, 13, 14],
        [4, 4, 4, 4], [6, 6, 6, 6], [6, 6, 6, 7],
        [6, 6, 7, 8], [6, 7, 8, 9], [11, 11, 11, 11],
        [1, 1, 1, 1], [1, 1, 1, 2], [6, 6, 6, 6],
        [6, 6, 6, 7], [11, 11, 11, 11], [11, 11, 11, 12],
    ])
    assert np.allclose(buf4[indice].obs_next[..., 0], [
        [11, 11, 12, 13], [11, 12, 13, 14], [11, 12, 13, 14],
        [4, 4, 4, 4], [6, 6, 6, 7], [6, 6, 7, 8],
        [6, 7, 8, 9], [6, 7, 8, 9], [11, 11, 11, 12],
        [1, 1, 1, 2], [1, 1, 1, 2], [6, 6, 6, 7],
        [6, 6, 6, 7], [11, 11, 11, 12], [11, 11, 11, 12],
    ])
    assert np.all(buf4.done == buf5.done)
    indice = buf5.sample_index(0)
    assert np.allclose(sorted(indice), [2, 7])
    assert np.all(np.isin(buf5.sample_index(100), indice))
    # manually change the stack num
    buf5.stack_num = 2
    for buf in buf5.buffers:
        buf.stack_num = 2
    indice = buf5.sample_index(0)
    assert np.allclose(sorted(indice), [0, 1, 2, 5, 6, 7, 10, 15, 20])
    batch, _ = buf5.sample(0)
    assert np.allclose(buf5[np.arange(buf5.maxsize)].weight, 1)
    buf5.update_weight(indice, batch.weight * 0)
    weight = buf5[np.arange(buf5.maxsize)].weight
    modified_weight = weight[[0, 1, 2, 5, 6, 7]]
    assert modified_weight.min() == modified_weight.max()
    assert modified_weight.max() < 1
    unmodified_weight = weight[[3, 4, 8]]
    assert unmodified_weight.min() == unmodified_weight.max()
    assert unmodified_weight.max() < 1
    cached_weight = weight[9:]
    assert cached_weight.min() == cached_weight.max() == 1
    # test Atari with CachedReplayBuffer, save_only_last_obs + ignore_obs_next
    buf6 = CachedReplayBuffer(
        ReplayBuffer(bufsize, stack_num=stack_num,
                     save_only_last_obs=True, ignore_obs_next=True),
        cached_num, size)
    obs = np.random.rand(size, 4, 84, 84)
    buf6.add(obs=[obs[2], obs[0]], act=[1, 1], rew=[0, 0], done=[0, 1],
             obs_next=[obs[3], obs[1]], cached_buffer_ids=[1, 2])
    assert buf6.obs.shape == (buf6.maxsize, 84, 84)
    assert np.allclose(buf6.obs[0], obs[0, -1])
    assert np.allclose(buf6.obs[14], obs[2, -1])
    assert np.allclose(buf6.obs[19], obs[0, -1])
    assert buf6[0].obs.shape == (4, 84, 84)


def test_multibuf_hdf5():
    size = 100
    buffers = {
        "vector": ReplayBufferManager([ReplayBuffer(size) for i in range(4)]),
        "cached": CachedReplayBuffer(ReplayBuffer(size), 4, size)
    }
    buffer_types = {k: b.__class__ for k, b in buffers.items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    info_t = torch.tensor([1.]).to(device)
    for i in range(4):
        kwargs = {
            'obs': Batch(index=np.array([i])),
            'act': i,
            'rew': np.array([1, 2]),
            'done': i % 3 == 2,
            'info': {"number": {"n": i, "t": info_t}, 'extra': None},
        }
        buffers["vector"].add(**Batch.stack([kwargs, kwargs, kwargs]),
                              buffer_ids=[0, 1, 2])
        buffers["cached"].add(**Batch.stack([kwargs, kwargs, kwargs]),
                              cached_buffer_ids=[0, 1, 2])

    # save
    paths = {}
    for k, buf in buffers.items():
        f, path = tempfile.mkstemp(suffix='.hdf5')
        os.close(f)
        buf.save_hdf5(path)
        paths[k] = path

    # load replay buffer
    _buffers = {k: buffer_types[k].load_hdf5(paths[k]) for k in paths.keys()}

    # compare
    for k in buffers.keys():
        assert len(_buffers[k]) == len(buffers[k])
        assert np.allclose(_buffers[k].act, buffers[k].act)
        assert _buffers[k].stack_num == buffers[k].stack_num
        assert _buffers[k].maxsize == buffers[k].maxsize
        assert np.all(_buffers[k]._indices == buffers[k]._indices)
    # check shallow copy in ReplayBufferManager
    for k in ["vector", "cached"]:
        buffers[k].info.number.n[0] = -100
        assert buffers[k].buffers[0].info.number.n[0] == -100
    # check if still behave normally
    for k in ["vector", "cached"]:
        kwargs = {
            'obs': Batch(index=np.array([5])),
            'act': 5,
            'rew': np.array([2, 1]),
            'done': False,
            'info': {"number": {"n": i}, 'Timelimit.truncate': True},
        }
        buffers[k].add(**Batch.stack([kwargs, kwargs, kwargs, kwargs]))
        act = np.zeros(buffers[k].maxsize)
        if k == "vector":
            act[np.arange(5)] = np.array([0, 1, 2, 3, 5])
            act[np.arange(5) + size] = np.array([0, 1, 2, 3, 5])
            act[np.arange(5) + size * 2] = np.array([0, 1, 2, 3, 5])
            act[size * 3] = 5
        elif k == "cached":
            act[np.arange(9)] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
            act[np.arange(3) + size] = np.array([3, 5, 2])
            act[np.arange(3) + size * 2] = np.array([3, 5, 2])
            act[np.arange(3) + size * 3] = np.array([3, 5, 2])
            act[size * 4] = 5
        assert np.allclose(buffers[k].act, act)

    for path in paths.values():
        os.remove(path)


if __name__ == '__main__':
    test_replaybuffer()
    test_ignore_obs_next()
    test_stack()
    test_segtree()
    test_priortized_replaybuffer()
    test_priortized_replaybuffer(233333, 200000)
    test_update()
    test_pickle()
    test_hdf5()
    test_replaybuffermanager()
    test_cachedbuffer()
    test_multibuf_stack()
    test_multibuf_hdf5()
