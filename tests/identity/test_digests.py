from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from ruleforge.judges.weights import MANIFEST_FNAME, WEIGHTS_DIGEST_VERSION, weights_digest
from ruleforge.core.errors import SamplingDesignError
from ruleforge.judges.weights import adapter_identity, model_weights_identity
from ruleforge.judges import anthropic_batch, batch_api
from ruleforge.judges.batch_api import BatchRequest
from tests.identity import published as pub

WEIGHTS_SHA256 = "213f11a1ef3b776dd3e13b8ccbe6a9b9c1f923a9250b09968de360bfbd442037"
ADAPTER_BYTES = b"tiny adapter weights\n"
ADAPTER_CONFIG = b'{"r": 4, "lora_alpha": 8, "base_model_name_or_path": "tiny-base"}\n'


def _safetensors(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    """A safetensors file: a little-endian header length, the JSON header, then the tensors' bytes in order."""
    header, offset, data = {}, 0, b""
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
        data += payload
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + data


def _two_tensor_checkpoint(directory: Path, *, split: bool = False) -> Path:
    tensors = {
        "model.embed.weight": ("F32", [2, 2], struct.pack("<4f", 0.5, -1.0, 2.0, 0.25)),
        "model.head.bias": ("BF16", [2], bytes([0x80, 0x3F, 0x00, 0xC0])),
    }
    directory.mkdir(parents=True)
    if split:
        for index, (name, value) in enumerate(tensors.items()):
            (directory / f"model-0000{index + 1}-of-00002.safetensors").write_bytes(_safetensors({name: value}))
    else:
        (directory / "model.safetensors").write_bytes(_safetensors(tensors))
    return directory


def _with_manifest(directory: Path, weights_sha256: str) -> Path:
    manifest = {"weights_digest": WEIGHTS_DIGEST_VERSION, "kind": "merge",
                "merge": {"saved_dtype": "float32"}, "result": {"weights_sha256": weights_sha256}}
    (directory / MANIFEST_FNAME).write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_the_weights_digest_of_a_small_checkpoint_is_its_recorded_value(tmp_path):
    assert weights_digest(_two_tensor_checkpoint(tmp_path / "one")) == WEIGHTS_SHA256
    assert weights_digest(_two_tensor_checkpoint(tmp_path / "two", split=True)) == WEIGHTS_SHA256


def test_derived_weights_are_named_by_the_digest_their_manifest_records(tmp_path):
    directory = _with_manifest(_two_tensor_checkpoint(tmp_path / "merged"), WEIGHTS_SHA256)
    identity = model_weights_identity(directory)
    assert (identity.sha256, identity.digest_version, identity.saved_dtype) == (
        WEIGHTS_SHA256, WEIGHTS_DIGEST_VERSION, "float32")
    assert identity.manifest_sha256 == pub.sha256_bytes((directory / MANIFEST_FNAME).read_bytes())


def test_a_changed_tensor_byte_changes_the_digest_and_is_refused_against_the_manifest(tmp_path):
    directory = _with_manifest(_two_tensor_checkpoint(tmp_path / "merged"), WEIGHTS_SHA256)
    shard = directory / "model.safetensors"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0x01  # the last byte of the last tensor
    shard.write_bytes(bytes(data))
    assert weights_digest(directory) != WEIGHTS_SHA256
    with pytest.raises(SamplingDesignError, match="a shard was changed or replaced after the merge"):
        model_weights_identity(directory)


def test_a_tiny_adapter_is_named_by_its_two_files(tmp_path):
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(ADAPTER_BYTES)
    (checkpoint / "adapter_config.json").write_bytes(ADAPTER_CONFIG)
    identity = adapter_identity(checkpoint / "adapter_model.safetensors")
    assert identity.label == "checkpoint-7/adapter_model.safetensors"
    assert identity.sha256 == "ac8e70fcb974a01d637ef1e3332c5d41e3fbeb01e63db5dc1b1cd47fa222fb88"
    assert identity.config_sha256 == pub.sha256_bytes(ADAPTER_CONFIG)
    assert (identity.rank, identity.base_model) == (4, "tiny-base")


BATCH = {row["provider"]: row for row in pub.read_fixture(pub.BATCH)}


def _lines(row) -> list[str]:
    return [line for part in row["parts"] for line in part["lines"]]


def test_a_recorded_openai_batch_is_addressed_by_its_request_lines():
    row = BATCH["openai_batch"]
    requests = []
    for line in _lines(row):
        payload = json.loads(line)
        assert payload["url"] == row["endpoint"]
        request = BatchRequest(custom_id=payload["custom_id"], body=payload["body"])
        assert batch_api.request_line(request, row["endpoint"]) == line.encode("utf-8")
        requests.append(request)
    assert batch_api.fingerprint(requests, row["endpoint"]) == row["work_dir"]


def test_a_recorded_anthropic_batch_is_addressed_by_its_request_lines():
    row = BATCH["anthropic_batch"]
    requests = []
    for line in _lines(row):
        payload = json.loads(line)
        request = BatchRequest(custom_id=payload["custom_id"], body=payload["params"])
        assert anthropic_batch.request_line(request) == line.encode("utf-8")
        requests.append(request)
    assert anthropic_batch.fingerprint(requests) == row["work_dir"]


def test_every_recorded_batch_part_is_its_lines():
    for row in BATCH.values():
        for part in row["parts"]:
            assert pub.sha256_bytes("".join(part["lines"]).encode("utf-8")) == part["sha256"]
