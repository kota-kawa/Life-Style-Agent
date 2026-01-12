import sys
import types


def _install_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _DummyLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        return "{}"


class _DummyEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


class _DummyFAISS:
    def __init__(self, *args, **kwargs):
        pass


# Install LLM stubs to avoid external API dependencies during tests.
_install_stub("langchain_openai", ChatOpenAI=_DummyLLM)
_install_stub("langchain_google_genai", ChatGoogleGenerativeAI=_DummyLLM)
_install_stub("langchain_anthropic", ChatAnthropic=_DummyLLM)
_install_stub("langchain_groq", ChatGroq=_DummyLLM)

# Provide minimal langchain_community stubs used at import time.
community = _install_stub("langchain_community")
embeddings = _install_stub("langchain_community.embeddings", HuggingFaceEmbeddings=_DummyEmbeddings)
vectorstores = _install_stub("langchain_community.vectorstores", FAISS=_DummyFAISS)
community.embeddings = embeddings
community.vectorstores = vectorstores

# Guard against indirect imports relying on langchain_core.
core = _install_stub("langchain_core")
_install_stub("langchain_core.pydantic_v1", Field=lambda *a, **k: None)
core.pydantic_v1 = sys.modules["langchain_core.pydantic_v1"]
