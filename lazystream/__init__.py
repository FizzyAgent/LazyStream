import concurrent.futures
import copy
from typing import Callable, Generic, Iterator, TypeVar, Optional, List

A = TypeVar("A")
B = TypeVar("B")


class LazyStream(Generic[A]):
    """
    Lazy Stream that only evaluates the final results when needed
    """

    def __init__(self, generator: Callable[[], A]):
        self._generator = generator

    def __getitem__(self, index: int) -> A:
        """
        Use evaluate() instead to get a list of values
        """
        raise NotImplementedError("LazyStream does not support indexing")

    def __len__(self) -> int:
        raise NotImplementedError("LazyStream does not support len()")

    def __safe_iter(self, limit: Optional[int] = None) -> Iterator[A]:
        i = 0
        while limit is None or i < limit:
            try:
                i += 1
                yield self._generator()
            except StopIteration:
                break

    def __safe_next(self) -> A:
        return next(self.__safe_iter())

    def __iter__(self) -> Iterator[A]:
        return self.__safe_iter()

    def __next__(self) -> A:
        return self.__safe_next()

    @staticmethod
    def from_lambda(func: Callable[[], A]) -> "LazyStream[A]":
        return LazyStream(func)

    @staticmethod
    def from_iterator(iterator: Iterator[A]) -> "LazyStream[A]":
        return LazyStream(lambda: next(iterator))

    def evaluate(self, limit: Optional[int] = None) -> List[A]:
        output: List[A] = []
        for value in self.__safe_iter(limit):
            output.append(value)
        return output

    def par_evaluate(
        self, limit: int, executor: concurrent.futures.Executor
    ) -> List[A]:
        futures: List[concurrent.futures._base.Future[A]] = []
        while len(futures) < limit:
            futures.append(executor.submit(self._generator))
        output: List[A] = []
        for future in futures:
            try:
                output.append(future.result())
            except StopIteration:
                break
        return output

    def reduce(
        self, func: Callable[[B, A], B], accum: B, limit: Optional[int] = None
    ) -> B:
        accum = copy.deepcopy(accum)
        for value in self.__safe_iter(limit):
            accum = func(accum, value)
        return accum

    def map(self, func: Callable[[A], B]) -> "LazyStream[B]":
        return LazyStream(lambda: func(self.__safe_next()))

    def map_enumerate(self, func: Callable[[int, A], B]) -> "LazyStream[B]":
        def iterator() -> Iterator[B]:
            for i, value in enumerate(self.__safe_iter()):
                yield func(i, value)

        return LazyStream.from_iterator(iterator())

    def par_map(
        self, func: Callable[[A], B], executor: concurrent.futures.Executor
    ) -> "LazyStream[B]":
        return LazyStream(lambda: executor.submit(func, self.__safe_next()).result())

    def filter(self, predicate: Callable[[A], bool]) -> "LazyStream[A]":
        def iterator() -> Iterator[A]:
            for value in self.__safe_iter():
                if predicate(value):
                    yield value

        return LazyStream.from_iterator(iterator())

    def for_each(self, func: Callable[[A], None]) -> "LazyStream[A]":
        def iterator() -> Iterator[A]:
            for value in self.__safe_iter():
                func(value)
                yield value

        return LazyStream.from_iterator(iterator())
