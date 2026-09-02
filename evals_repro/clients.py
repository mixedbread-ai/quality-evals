import cohere
import voyageai
from google import genai
from google.genai import types
from mixedbread import Mixedbread


def voyage() -> voyageai.Client:
    return voyageai.Client(max_retries=12, timeout=300)


def cohere_v2() -> cohere.ClientV2:
    return cohere.ClientV2(timeout=120)


def mixedbread() -> Mixedbread:
    return Mixedbread(max_retries=8, timeout=120)


def gemini() -> genai.Client:
    retry = types.HttpRetryOptions(attempts=8, initial_delay=2, max_delay=60)
    return genai.Client(http_options=types.HttpOptions(timeout=300_000, retry_options=retry))
