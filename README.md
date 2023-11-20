# Lazy Stream

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Lazy Stream is a library for lazy evaluation in Python, inspired by the Scala Stream class.

Lazy Stream allows you to store and chain operations on a finite or infinite sequence of values. The final result will only be computed when needed, for better memory and performance over conventional Python lists.

## Installation

Lazy Stream can be installed via pip:

```bash
pip install lazystream
```

## Usage

Lazy Stream supports basic stream operations and parallelism. Below is a quick overview of the current features. Feel free to request additional features.

### Creation

Lazy Stream is very easy to use. You can create a stream from any function or iterator and obtain the results via `evaluate`. Finite and infinite streams are supported, but care must be taken when using infinite streams to avoid infinite operation.

```python
from lazystream import LazyStream

# Finite stream from iterator
incremental_stream = LazyStream.from_iterator(iter(range(5)))
incremental_stream.evaluate()
# [0, 1, 2, 3, 4]

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
        
# Infinite stream from function-defined iterator
fibo_stream = LazyStream.from_iterator(fibonacci())
fibo_stream.evaluate(5)
# [0, 1, 1, 2, 3]
fibo_stream.evaluate()
# Unsafe, will never terminate
```

You can also create a stream from a function, which will be evaluated lazily:

```python
import random
from lazystream import LazyStream

# Infinite streams from function
random_stream = LazyStream.from_lambda(lambda: random.randint(0, 100))
```

### Operations

You can chain operations on streams, which will be evaluated lazily. Classic stream-compatible operations are supported, such as `map` and `filter`. Non-stream-compatible operations are not supported and should be done after evaluation.

```python
from lazystream import LazyStream

stream = LazyStream.from_iterator(iter(range(10)))
stream.filter(lambda x: x % 2 == 0).map(lambda x: x * 2).evaluate()
# [0, 4, 8, 12, 16]
```

### Evaluation

You can obtain the results of a stream as a list via `evaluate`, which allows you to optionally set a limit on the number of elements to evaluate. This is useful/required for infinite streams.

```python
from lazystream import LazyStream

stream = LazyStream.from_lambda(lambda: 1)
stream.evaluate(5)
# [1, 1, 1, 1, 1]
```

You can also use the stream as an iterator itself. Note that proper termination conditions must be set for infinite streams.

```python
import random
from lazystream import LazyStream

# Iterate on a finite stream
finite_stream = LazyStream.from_iterator(iter(range(10)))
for x in finite_stream:
    print(x)

# Iterate on an infinite stream
infinite_stream = LazyStream.from_lambda(lambda: random.randint(0, 1))
for x in infinite_stream:
    print(x)
    if x == 1:
        break
```

In addition, the `reduce` operation is supported, which allows you to obtain a single value from a stream.

```python
from lazystream import LazyStream

# Reduce on a finite stream
stream = LazyStream.from_iterator(iter(range(10)))
stream.reduce(lambda x, y: x + y, accum=0)
# 45

# Reduce on an infinite stream
stream = LazyStream.from_lambda(lambda: 1)
stream.reduce(lambda x, y: x + y, accum=0, limit=5)
# 5
```

### Parallelism

You can add parallelism to the stream via functional mapping `par_map` and via results evaluation `par_evaluate`. Due to Python's parallelism implementation, this is only useful if your mapping function is IO-bound.

Note that Lazy Stream does not check for thread safety.

```python
from concurrent.futures import ThreadPoolExecutor
from lazystream import LazyStream

def io_bound_function(x):
    # Do some IO-bound operation
    return x

stream = LazyStream.from_iterator(iter(range(10)))
stream.par_map(
    io_bound_function, executor=ThreadPoolExecutor(4)
).evaluate()
```
