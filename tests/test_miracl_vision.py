import pyarrow as pa
import pyarrow.parquet as pq

from evals_repro.benchmarks.miracl_vision import BENCHMARK, build_subset


def write(root, name, rows):
    pq.write_table(pa.Table.from_pylist(rows), root / f"{name}.parquet")


def test_build_subset_maps_images_to_pages(tmp_path):
    write(tmp_path, "queries", [{"_id": "7", "text": "q"}, {"_id": "8", "text": "unjudged"}])
    write(
        tmp_path,
        "corpus",
        [
            {"_id": "10", "title": "B", "text": "b", "image_id": "i1"},
            {"_id": "2", "title": "A", "text": "a", "image_id": "i0"},
        ],
    )
    write(tmp_path, "qrels", [{"query-id": 7, "corpus-id": 10, "score": 1}])
    write(
        tmp_path,
        "images",
        [
            {"file_name": "i0", "image": {"bytes": b"A", "path": None}},
            {"file_name": "i1", "image": {"bytes": b"B", "path": None}},
        ],
    )
    subset = build_subset("yo", tmp_path)
    assert list(subset.pages) == ["2", "10"] and subset.native_language == "yoruba"
    assert [q.id for q in subset.queries] == ["7"] and subset.queries[0].language == "yoruba"
    assert subset.page_images() == [("A", b"A"), ("B", b"B")] and subset.pages["2"] == "title: A\na"
    assert not BENCHMARK.crosslingual and len(BENCHMARK.subsets) == 18
