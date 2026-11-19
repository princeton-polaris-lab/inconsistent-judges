from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from ruleforge.judges import tokenizer as tk

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_ROOT = os.environ.get("RULEFORGE_MODEL_ROOT") or str(Path.home() / "models")

#: tokenizer.json's pre-tokenizer regex in every Qwen3.5-class checkpoint.
QWEN35_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

QWEN35_9B_WEIGHTS = "Qwen3.5-9B"


def _checkpoint(path: Path, tokenizer_class: str | None) -> Path:
    """A synthetic checkpoint tokenizing like Qwen3.5; ``tokenizer_class`` as its tokenizer_config.json names it."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN35_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([tk.FIDELITY_PROBE] * 20, trainer)
    tokenizer.save(str(path / tk.TOKENIZER_JSON_FNAME))
    config = {
        "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None, "unk_token": None,
        "add_prefix_space": False, "model_max_length": 4096,
    }
    if tokenizer_class is not None:
        config["tokenizer_class"] = tokenizer_class
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    return path


@pytest.fixture(scope="module")
def qwen35_like(tmp_path_factory) -> Path:
    """Names Qwen2Tokenizer, as every Qwen3.5-class checkpoint does."""
    return _checkpoint(tmp_path_factory.mktemp("qwen35_like"), "Qwen2Tokenizer")


@pytest.fixture(scope="module")
def faithful(tmp_path_factory) -> Path:
    """The same tokenizer.json naming no class: AutoTokenizer takes it verbatim."""
    return _checkpoint(tmp_path_factory.mktemp("faithful"), None)


@pytest.fixture
def auto_registries():
    """Undo what vLLM registers with transformers' Auto classes."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    saved = [(mapping, dict(mapping._extra_content)) for mapping in (CONFIG_MAPPING, TOKENIZER_MAPPING)]
    yield
    for mapping, extra in saved:
        mapping._extra_content.clear()
        mapping._extra_content.update(extra)


def reference(path: Path, text: str = tk.FIDELITY_PROBE) -> list[int]:
    ids = tk.reference_token_ids(path, text)
    assert ids is not None
    return ids


def _auto(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def test_autotokenizer_splits_combining_marks_apart_from_tokenizer_json(qwen35_like):
    """The defect the guard exists for (transformers #49066)."""
    loaded = _auto(qwen35_like)
    assert type(loaded).__name__ == "Qwen2Tokenizer"
    ids = loaded.encode(tk.FIDELITY_PROBE, add_special_tokens=False)
    assert ids != reference(qwen35_like) and len(ids) > len(reference(qwen35_like))
    # Latin text is untouched: only the \p{M} branches differ
    assert loaded.encode("It's 2026", add_special_tokens=False) == reference(qwen35_like, "It's 2026")


def test_a_tokenizer_that_is_the_published_one_passes(faithful, qwen35_like):
    tk.check_tokenizer_fidelity(_auto(faithful), faithful, where="HF")
    from transformers import TokenizersBackend

    verbatim = TokenizersBackend.from_pretrained(str(qwen35_like))
    for text in (tk.FIDELITY_PROBE, "สวัสดีครับ", "नमस्ते", "مَرْحَبًا"):
        assert verbatim.encode(text, add_special_tokens=False) == reference(qwen35_like, text), text
    tk.check_tokenizer_fidelity(verbatim, qwen35_like, where="HF")


def test_fidelity_refuses_the_rebuilt_regex_and_names_the_cause(qwen35_like):
    with pytest.raises(tk.TokenizerMismatch) as error:
        tk.check_tokenizer_fidelity(_auto(qwen35_like), qwen35_like, where="vLLM engine")
    message = str(error.value)
    assert "the vLLM engine tokenizer (Qwen2Tokenizer)" in message
    assert f"as {len(reference(qwen35_like))} (first difference" in message
    assert "#49066" in message and "blocked on this stack" in message


def test_the_first_difference_is_reported_even_when_one_is_a_prefix(qwen35_like):
    class _Short:
        def encode(self, text, add_special_tokens=False):
            return reference(qwen35_like)[:-1]

    with pytest.raises(tk.TokenizerMismatch, match=rf"first difference at token {len(reference(qwen35_like)) - 1}"):
        tk.check_tokenizer_fidelity(_Short(), qwen35_like, where="HF")


def test_a_checkpoint_without_tokenizer_json_is_not_checked(tmp_path, caplog):
    class _Never:
        def encode(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("encoded without a reference")

    assert not tk.has_tokenizer_json(tmp_path) and tk.reference_token_ids(tmp_path) is None
    with caplog.at_level("INFO", logger=tk.__name__):
        tk.check_tokenizer_fidelity(_Never(), tmp_path, where="HF")
    assert "no tokenizer.json" in caplog.text


@pytest.fixture
def fake_llm(monkeypatch):
    """A stand-in ``vllm.LLM`` serving ``served["tokenizer"]``; returns (the kwargs it saw, served)."""
    seen: dict = {}
    served: dict = {}

    class _LLM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def get_tokenizer(self):
            return served["tokenizer"]

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_LLM))
    return seen, served


def test_engine_start_checks_both_tokenizers_and_serves_a_faithful_one(fake_llm, faithful):
    from ruleforge.judges.models import _ThinkingTokenizer, init_local_model

    seen, served = fake_llm
    served["tokenizer"] = _auto(faithful)
    _, tokenizer = init_local_model(
        model_path=str(faithful), tp_size=1, enable_thinking=True, attention_backend="TRITON_ATTN"
    )
    assert isinstance(tokenizer, _ThinkingTokenizer) and seen["model"] == str(faithful)
    assert tokenizer.eos_token == "<|im_end|>"
    assert seen["attention_config"] == {"backend": "TRITON_ATTN"}
    text = "<|im_start|>user\nhi<|im_end|>"
    expected = _auto(faithful)(text, add_special_tokens=False)["input_ids"]
    assert tokenizer(text, add_special_tokens=False)["input_ids"] == expected


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_the_engine_s_tokenizer_passes_the_configured_thinking_to_the_chat_template(fake_llm, faithful,
                                                                                   enable_thinking):
    """A template that thinks unless told otherwise (Qwen3's) needs a real False, not a missing argument."""
    from ruleforge.judges.models import init_local_model

    seen, served = fake_llm
    served["tokenizer"] = _auto(faithful)
    _, tokenizer = init_local_model(model_path=str(faithful), tp_size=1, enable_thinking=enable_thinking)
    sent = {}
    tokenizer._tokenizer.apply_chat_template = lambda *args, **kwargs: sent.update(kwargs)
    tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    assert sent["enable_thinking"] is enable_thinking


def test_engine_start_refuses_an_engine_tokenizer_that_is_not_tokenizer_json(fake_llm, faithful, qwen35_like):
    from ruleforge.judges.models import init_local_model

    _, served = fake_llm
    served["tokenizer"] = _auto(qwen35_like)
    with pytest.raises(tk.TokenizerMismatch, match=r"^the vLLM engine tokenizer \(Qwen2Tokenizer\)"):
        init_local_model(model_path=str(faithful), tp_size=1)


def test_engine_start_refuses_a_qwen35_class_checkpoint_before_building_the_engine(fake_llm, qwen35_like):
    from ruleforge.judges.models import init_local_model

    seen, _ = fake_llm
    with pytest.raises(tk.TokenizerMismatch, match=r"^the HF tokenizer \(Qwen2Tokenizer\).*#49066"):
        init_local_model(model_path=str(qwen35_like), tp_size=1)
    assert seen == {}


def test_an_engine_that_fails_to_start_is_named_as_such(monkeypatch, faithful):
    from ruleforge.judges.models import init_local_model

    def no_engine(**kwargs):
        raise ValueError("no GPU")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=no_engine))
    with pytest.raises(RuntimeError, match="^Failed to initialize local model: no GPU$") as failed:
        init_local_model(model_path=str(faithful), tp_size=1)
    assert isinstance(failed.value.__cause__, ValueError)


def test_judges_check_refuses_a_misread_checkpoint_in_one_line_before_its_engine_is_built(monkeypatch, capsys,
                                                                                          fake_llm, qwen35_like):
    import ruleforge.judges.models as models_module
    from ruleforge.cli.app import main

    seen, _ = fake_llm
    monkeypatch.setattr(models_module, "engine_argument_names", frozenset)
    monkeypatch.setattr("ruleforge.judges.providers.require", lambda purpose, *modules: None)
    assert main(["judges", "check", f"local:{qwen35_like}", "--probe"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ruleforge judges: TokenizerMismatch: the HF tokenizer (Qwen2Tokenizer) encodes")
    assert len(err.splitlines()) == 1 and "Traceback" not in err and seen == {}


def test_vllms_default_tokenizer_has_the_defect_and_is_refused(qwen35_like, faithful, auto_registries):
    registry = pytest.importorskip("vllm.tokenizers.registry", reason="needs the vllm group")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    served = registry.get_tokenizer(str(qwen35_like))
    with pytest.raises(tk.TokenizerMismatch):
        tk.check_tokenizer_fidelity(served, qwen35_like, where="vLLM engine")
    tk.check_tokenizer_fidelity(registry.get_tokenizer(str(faithful)), faithful, where="vLLM engine")
    # what the fixture undoes: vLLM registered its own qwen3_5 config class
    assert "qwen3_5" in CONFIG_MAPPING._extra_content


def test_the_auto_registries_are_restored(auto_registries):
    """Runs after the vLLM test in file order: transformers' own class again."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert "qwen3_5" not in CONFIG_MAPPING._extra_content
    assert CONFIG_MAPPING["qwen3_5"].__module__.startswith("transformers.")


QWEN35_9B_PINNED = {
    "สวัสดีครับ": [34469, 168607, 153295],
    "नमस्ते दुनिया": [58069, 84237, 150104, 153348, 184642, 235886],
    "مَرْحَبًا": [9873, 164865, 28850, 150765, 150021],
    "Hello, world!": [9419, 11, 1814, 0],
}


def _local_model_configs():
    from omegaconf import OmegaConf

    configs = []
    for path in sorted((REPO_ROOT / "configs" / "model").glob("*.yaml")):
        cfg = OmegaConf.load(path)
        if cfg.get("model_type") == "local":
            configs.append((path.name, cfg))
    return configs


def _fidelity(tokenizer, weights) -> bool:
    try:
        tk.check_tokenizer_fidelity(tokenizer, weights, where="probe")
    except tk.TokenizerMismatch:
        return False
    return True


def _encodes_pinned(tokenizer) -> bool:
    return all(tokenizer.encode(text, add_special_tokens=False) == ids for text, ids in QWEN35_9B_PINNED.items())


@pytest.mark.cluster
def test_qwen35_9b_is_refused_exactly_while_it_is_misread(monkeypatch, auto_registries):
    weights = Path(REAL_MODEL_ROOT) / QWEN35_9B_WEIGHTS
    if not tk.has_tokenizer_json(weights):
        pytest.skip(f"no {weights}")
    for text, ids in QWEN35_9B_PINNED.items():
        assert tk.reference_token_ids(weights, text) == ids, text
    hf = _auto(weights)
    assert _fidelity(hf, weights) is _encodes_pinned(hf)
    registry = pytest.importorskip("vllm.tokenizers.registry", reason="needs the vllm group")
    served = registry.get_tokenizer(str(weights), trust_remote_code=True)
    assert _fidelity(served, weights) is _encodes_pinned(served)


@pytest.mark.cluster
def test_every_local_judge_config_is_served(monkeypatch, auto_registries):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", REAL_MODEL_ROOT)
    try:
        from vllm.tokenizers.registry import get_tokenizer
    except ImportError:  # pragma: no cover - the vllm group is absent
        get_tokenizer = None
    passed = {}
    for name, cfg in _local_model_configs():
        weights = Path(str(cfg.model_path))
        if not tk.has_tokenizer_json(weights):
            continue
        verdicts = [_fidelity(_auto(weights), weights)]
        if get_tokenizer is not None:
            verdicts.append(_fidelity(get_tokenizer(str(weights), trust_remote_code=True), weights))
        passed[name] = verdicts
    if not passed:
        pytest.skip(f"no local model weights under {REAL_MODEL_ROOT}")
    assert "qwen3-8b.yaml" in passed
    assert {name: verdicts for name, verdicts in passed.items() if not all(verdicts)} == {}
