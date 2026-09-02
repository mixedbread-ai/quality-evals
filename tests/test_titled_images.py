from evals_repro.cache import digest
from evals_repro.embedders import cohere, gemini, voyage_multimodal
from evals_repro.embedders.images import data_url

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + (100).to_bytes(4, "big") + (200).to_bytes(4, "big")


def test_title_precedes_image_for_every_provider():
    assert voyage_multimodal.image_input("T", PNG)["content"][0] == {"type": "text", "text": "title: T"}
    assert cohere.image_input("T", PNG)["content"][0] == {"type": "text", "text": "title: T"}
    assert gemini.image_content("T", PNG).parts[0].text == "title: T"
    assert [p["type"] for p in voyage_multimodal.image_input("", PNG)["content"]] == ["image_base64"]
    assert cohere.image_input("", PNG)["content"] == [{"type": "image_url", "image_url": {"url": data_url(PNG)}}]
    assert gemini.image_content("", PNG).parts[0].inline_data.data == PNG


def test_voyage_counts_title_tokens():
    assert voyage_multimodal.image_tokens("abcdef", PNG) == voyage_multimodal.image_tokens("", PNG) + 2


def test_digest_separates_title_from_image():
    assert digest([("a", b"b")]) != digest([("ab", b"")]) != digest(["ab"])
