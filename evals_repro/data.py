from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

Content = Literal["markdown", "images"]
Setting = Literal["monolingual", "crosslingual"]
Images = Callable[[], Iterator[tuple[str, bytes]]]
TitledImage = tuple[str, bytes]


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    language: str


@dataclass
class Subset:
    name: str
    source: str
    native_language: str
    pages: dict[str, str]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]
    images: Images = field(default=lambda: iter(()), repr=False)
    titles: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def languages(self) -> list[str]:
        return sorted({q.language for q in self.queries})

    def queries_in(self, language: str) -> list[Query]:
        return [q for q in self.queries if q.language == language]

    def qrels_for(self, queries: Iterable[Query]) -> dict[str, dict[str, int]]:
        return {q.id: self.qrels[q.id] for q in queries}

    def title(self, cid: str) -> str:
        return self.titles.get(cid, "")

    def page_images(self) -> list[TitledImage]:
        images = dict(self.images())
        return [(self.title(cid), images[cid]) for cid in self.pages]


def check_consistency(subset: Subset) -> None:
    missing = {cid for rels in subset.qrels.values() for cid in rels} - subset.pages.keys()
    assert not missing, f"{subset.name}: {len(missing)} judged pages absent from corpus"
    ids = [q.id for q in subset.queries]
    assert len(ids) == len(set(ids)), f"{subset.name}: duplicate query ids"
    assert set(ids) == subset.qrels.keys(), f"{subset.name}: queries and qrels disagree"
    per_language = {lang: len(subset.queries_in(lang)) for lang in subset.languages}
    assert len(set(per_language.values())) == 1, f"{subset.name}: uneven languages {per_language}"
