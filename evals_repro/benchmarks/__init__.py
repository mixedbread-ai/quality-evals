from collections.abc import Callable
from dataclasses import dataclass

from evals_repro.data import Subset


@dataclass(frozen=True)
class Benchmark:
    name: str
    subsets: tuple[str, ...]
    languages: tuple[str, ...]
    load: Callable[[str], Subset]

    @property
    def crosslingual(self) -> bool:
        return bool(self.languages)


def registry() -> dict[str, Benchmark]:
    from evals_repro.benchmarks import miracl_vision, vidore_v3

    return {b.name: b for b in (vidore_v3.BENCHMARK, miracl_vision.BENCHMARK)}
