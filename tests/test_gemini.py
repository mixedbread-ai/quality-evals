from types import SimpleNamespace

from evals_repro.embedders.gemini import GeminiEmbedder


class FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents):
        self.calls.append([c.parts[0].text or c.parts[0].inline_data.mime_type for c in contents])
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(c.parts))]) for c in contents])


def test_each_text_and_image_is_its_own_content_with_documented_prompts():
    models = FakeModels()
    embedder = GeminiEmbedder(client=SimpleNamespace(models=models), max_batch_items=2, workers=1)
    texts = embedder.embed(["a", "", "b"], "query")
    docs = embedder.embed(["d"], "document")
    images = embedder.embed_images([("", b"\x89PNG"), ("t", b"\x89PNG"), ("", b"\x89PNG")])
    assert texts[:, 0].tolist() == [1, 0, 1] and docs.shape == (1, 1) and images.shape == (3, 1)
    assert models.calls[0] == ["task: search result | query: a", "task: search result | query: b"]
    assert models.calls[1] == ["title: none | text: d"]
    assert models.calls[2] == ["image/png", "title: t"] and models.calls[3] == ["image/png"]


def test_noprompt_variant_sends_bare_text():
    from evals_repro.embedders.gemini import BARE

    models = FakeModels()
    embedder = GeminiEmbedder(client=SimpleNamespace(models=models), prompts=BARE, workers=1)
    embedder.embed(["a"], "query")
    embedder.embed(["d"], "document")
    assert embedder.name == "gemini-embedding-2-noprompt" and models.calls == [["a"], ["d"]]
