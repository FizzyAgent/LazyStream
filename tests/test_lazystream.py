import random
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest

from lazystream import LazyStream

THREADPOOL = ThreadPoolExecutor(4)


def test_laziness():
    """
    Test that the generator is only called when needed
    """
    x = random.randint(0, 100)
    stream = LazyStream.from_lambda(lambda: x)
    for _ in range(10):
        # The return value should be the same as the current value of x
        x = random.randint(0, 100)
        for _ in range(10):
            assert next(stream) == x


def test_iterate():
    stream = LazyStream.from_iterator(iter(range(20)))
    i = 0
    x = 0
    for i, x in enumerate(stream):
        assert x == i
    assert x == i == 19


def test_next():
    stream = LazyStream.from_iterator(iter(range(20)))
    for i in range(20):
        x = next(stream)
        assert x == i
    with pytest.raises(StopIteration):
        # The only case we expect to catch StopIteration
        next(stream)


def test_evaluate():
    stream = LazyStream.from_iterator(iter(range(20)))
    # evaluate should return a list of the first 10 elements
    res = stream.evaluate(10)
    assert res == list(range(10))
    # evaluate should terminate early if the generator runs out
    res = stream.evaluate(100)
    assert res == list(range(10, 20))
    # evaluate should be empty if generator runs out
    res = stream.evaluate(10)
    assert res == []

    # par_evaluate should behave the same
    stream = LazyStream.from_iterator(iter(range(20)))
    res = stream.par_evaluate(10, executor=THREADPOOL)
    assert res == list(range(10))
    res = stream.par_evaluate(100, executor=THREADPOOL)
    assert res == list(range(10, 20))
    res = stream.par_evaluate(10, executor=THREADPOOL)
    assert res == []


def test_reduce():
    stream = LazyStream.from_iterator(iter(range(100)))

    res = stream.reduce(func=lambda x, y: x + y, accum=0, limit=10)
    assert res == sum(range(10))

    res = stream.reduce(func=lambda x, y: x + y, accum=0, limit=30)
    assert res == sum(range(10, 40))

    res = stream.reduce(func=lambda x, y: x + y, accum=0, limit=50)
    assert res == sum(range(40, 90))


def test_map():
    stream = LazyStream.from_iterator(iter(range(50)))

    mapped = stream.map(lambda x: x + 1)
    assert mapped.evaluate(10) == list(range(1, 11))
    assert mapped.evaluate(10) == list(range(11, 21))

    par_mapped = stream.par_map(lambda x: x + 1, executor=THREADPOOL)
    assert par_mapped.evaluate(10) == list(range(21, 31))
    assert par_mapped.evaluate(10) == list(range(31, 41))
    # evaluate should terminate safely if the generator runs out
    assert par_mapped.evaluate(100) == list(range(41, 51))
    # evaluate should be empty if generator runs out
    assert par_mapped.evaluate(10) == []

    def error_mapper(_: int) -> int:
        raise NotImplementedError("This should not be called")

    stream = LazyStream.from_iterator(iter(range(50)))
    # Should not raise an error
    error_mapped = stream.map(error_mapper)
    # Should raise an error when called
    with pytest.raises(NotImplementedError):
        error_mapped.evaluate(10)


def test_flatten():
    x: List[int] = []

    def gen() -> List[int]:
        x.append(len(x))
        return x

    stream = LazyStream.from_lambda(gen)
    flattened = stream.flatten()
    assert flattened.evaluate(10) == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]


def test_filter():
    stream = LazyStream.from_lambda(lambda: random.randint(0, 1))

    filtered = stream.filter(lambda x: x > 0)
    assert filtered.evaluate(10) == [1] * 10

    def error_filter(_: int) -> bool:
        raise NotImplementedError("This should not be called")

    # Should not raise an error
    error_filtered = stream.filter(error_filter)
    # Should raise an error when called
    with pytest.raises(NotImplementedError):
        error_filtered.evaluate(10)


def test_flatten_option():
    stream = LazyStream.from_lambda(lambda: random.choice((None, 1)))
    flattened = stream.flatten_option()
    assert flattened.evaluate(10) == [1] * 10


def test_distinct():
    stream = LazyStream.from_lambda(lambda: random.randint(0, 1))
    distinct = stream.distinct().evaluate(2)
    assert len(distinct) == 2
    assert set(distinct) == {0, 1}

    stream = LazyStream.from_lambda(lambda: random.randint(0, 4))
    distinct = stream.distinct_by(lambda x: x % 2).evaluate(2)
    assert len(distinct) == 2
    assert set([x % 2 for x in distinct]) == {0, 1}


def test_zip():
    x = 1
    s1 = LazyStream.from_iterator(iter(range(100)))
    s2 = LazyStream.from_lambda(lambda: x)
    zipped = s1.zip(s2)

    # Test finite evaluation
    assert zipped.evaluate(10) == list(zip(range(10), [1] * 10))
    # Test laziness
    x = 2
    assert zipped.evaluate(10) == list(zip(range(10, 20), [2] * 10))
    # Test zipping finite and infinite streams
    assert zipped.evaluate()[-1] == (99, 2)
