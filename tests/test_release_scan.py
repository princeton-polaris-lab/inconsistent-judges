from __future__ import annotations

import gzip
import io
import json
import os
import struct
import subprocess
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest

from tests.support import leak_scan as scan
from tests.support.leak_scan import (
    Allowance,
    Hit,
    Policy,
    ReleaseError,
    Scanner,
    ThirdParty,
    TreeAllowance,
    TreePolicy,
    forbidden_reason,
    scan_dir,
    scan_tree,
    unsafe_name,
)

STRICT = ("zorblax", "q-lab-77", "4815162342")
BROAD = ("frobnitz", "wug corp")
POLICY = Policy(strict=STRICT, broad=BROAD)


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _scan(files, policy=POLICY):
    scanner = Scanner(policy)
    for path, data in files.items():
        scanner.scan_file(path, data)
    return scanner.result()


def _terms(result):
    return sorted((hit.path, hit.term) for hit in result.hits)


def test_a_strict_term_in_code_fails_case_insensitively_as_a_substring():
    result = _scan({"src/a.py": b"# first line\nvalue = 'ZorblaxLabs'\n"})
    assert not result.ok
    (hit,) = result.hits
    assert (hit.path, hit.term, hit.kind, hit.view, hit.line, hit.match) == (
        "src/a.py", "zorblax", "strict", "text", 2, "Zorblax")
    assert "ZorblaxLabs" in hit.context


def test_an_all_digit_term_does_not_match_inside_a_longer_number():
    text = "id 4815162342, pi 948151623420, x4815162342y"
    result = _scan({"a.txt": text.encode()})
    assert [hit.offset for hit in result.hits] == [text.index("4815162342"), text.index("x4815162342y") + 1]


def test_a_broad_term_is_word_bounded_with_underscore_and_hyphen_as_boundaries():
    result = _scan({"a.txt": b"frobnitz_dir frobnitzer xfrobnitz Frobnitz-1 path/frobnitz"})
    assert [hit.match for hit in result.hits] == ["frobnitz", "Frobnitz", "frobnitz"]
    assert {hit.kind for hit in result.hits} == {"broad"}


def test_a_space_in_a_term_matches_runs_of_separators_but_not_nothing():
    result = _scan({"a.txt": b"wug corp; wug-corp; WUG.Corp; wug__corp; wug%20corp; wugcorp"})
    assert [hit.match for hit in result.hits] == ["wug corp", "wug-corp", "WUG.Corp", "wug__corp", "wug%20corp"]


def test_unicode_lookalikes_read_as_the_ascii_character_and_hits_record_the_file_text():
    text = "Q\u2011Lab\u201177, q\u2212lab\u221277, wug\u00a0corp, wug\u2013corp, wug\u2009-\u202fcorp"
    result = _scan({"a.md": text.encode(), "q\u2010lab\u201077.png": b""})
    assert [(hit.path, hit.view, hit.match) for hit in result.hits] == [
        ("a.md", "text", "Q\u2011Lab\u201177"), ("a.md", "text", "q\u2212lab\u221277"),
        ("a.md", "text", "wug\u00a0corp"), ("a.md", "text", "wug\u2013corp"),
        ("a.md", "text", "wug\u2009-\u202fcorp"), ("q\u2010lab\u201077.png", "name", "q\u2010lab\u201077")]
    assert _terms(_scan({"a.md": b"Q-Lab and wug-corp"}, Policy(("q\u2013lab",), ("wug\u00a0corp",)))) == [
        ("a.md", "q\u2013lab"), ("a.md", "wug\u00a0corp")]
    key = "sk\u2011proj\u2011" + "a1B2" * 8
    assert scan.redact_secrets(f"key={key}") == "key=" + scan.SECRET_MASK * len(key)
    assert _terms(_scan({"a.py": key.encode()})) == [("a.py", "secret:openai-style-key")]


def test_case_is_ignored_for_ascii_letters_only_and_the_probe_agrees_with_the_pattern():
    sigma = Policy(("\u0391\u03a3",))
    assert _terms(_scan({"a.md": "\u0391\u03a3\u0391".encode()}, sigma)) == [("a.md", "\u0391\u03a3")]
    assert _scan({"a.md": "\u03b1\u03c3\u03b1".encode()}, sigma).ok
    assert _terms(_scan({"a.md": b"ZORBLAX"}, Policy(("ZorBlax",)))) == [("a.md", "ZorBlax")]


def test_a_clean_tree_passes():
    result = _scan({"README.md": b"# Nothing to see\n", "src/m.py": b"x = 1\n"})
    assert result.ok and result.hits == [] and result.files == 2


def test_a_binary_file_is_read_as_latin1():
    result = _scan({"blob.bin": b"\x00\x01\xffzorblax\x00"})
    (hit,) = result.hits
    assert hit.view == "latin-1" and hit.term == "zorblax"


@pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be"])
def test_a_utf16_text_file_is_read_with_or_without_its_byte_order_mark(codec):
    text = "notes by Zorblax\n"
    (hit,) = _scan({"notes.txt": text.encode(codec)}).hits
    assert (hit.view, hit.term, hit.match) == ("utf-16", "zorblax", "Zorblax")


@pytest.mark.parametrize("data", [b"\x00z\x00o\x00r", b"z\x00o\x00r\x00b\x00l\x00a\x00x\x01\x01"])
def test_short_or_mixed_nul_data_is_not_read_as_utf16(data):
    assert "utf-16" not in {view for view, _ in scan.views(data)}


DATA = b'{"prompt": "frobnitz and wug corp"}\n'


def test_a_broad_term_passes_in_third_party_data_while_a_strict_term_fails_there():
    data = DATA + b'{"prompt": "zorblax"}\n'
    policy = Policy(STRICT, BROAD, third_party=(ThirdParty("data/x.jsonl", _sha(data), "CC0-1.0"),))
    result = _scan({"data/x.jsonl": data}, policy)
    assert _terms(result) == [("data/x.jsonl", "zorblax")]
    assert result.third_party == {"applied": ["data/x.jsonl"], "lapsed": [], "unused": []}


def test_a_third_party_sha_mismatch_revokes_the_allowance():
    policy = Policy(STRICT, BROAD, third_party=(ThirdParty("data/x.jsonl", "0" * 64, "CC0-1.0"),))
    result = _scan({"data/x.jsonl": DATA}, policy)
    assert _terms(result) == [("data/x.jsonl", "frobnitz"), ("data/x.jsonl", "wug corp")]
    assert result.third_party["lapsed"] == ["data/x.jsonl"]


def test_first_party_only_rules_skip_third_party_data_and_the_rest_do_not():
    home = "/home/" + "alice"
    email = "alice" + "@" + "corp.io"
    key = "AK" + "IA" + "Q" * 16
    data = f"{home} {email} {key}".encode()
    policy = Policy(STRICT, third_party=(ThirdParty("d.jsonl", _sha(data), "MIT"),))
    assert _terms(_scan({"d.jsonl": data}, policy)) == [("d.jsonl", "secret:aws-access-key-id")]
    assert _terms(_scan({"d.jsonl": data})) == [
        ("d.jsonl", "email"), ("d.jsonl", "path:/home"), ("d.jsonl", "secret:aws-access-key-id")]


def test_the_module_names_no_identifier_of_its_own():
    assert {rule.term for rule in scan.RULES if not rule.term.startswith("secret:")} == {
        "path:/home", "path:/Users", "email"}


def test_an_exception_pinned_by_sha256_excuses_one_term_in_one_file():
    data = b"zorblax and frobnitz\n"
    policy = Policy(STRICT, BROAD, exceptions=(Allowance("a.md", _sha(data), "zorblax", "a quoted name"),))
    result = _scan({"a.md": data}, policy)
    assert _terms(result) == [("a.md", "frobnitz")]
    assert [hit.term for hit in result.excepted] == ["zorblax"]
    assert result.exceptions_unused == []


def test_an_exception_lapses_when_the_file_changes_and_unused_entries_are_listed():
    stale = Allowance("a.md", "0" * 64, "zorblax", "was justified once")
    unused = Allowance("gone.md", "1" * 64, "zorblax", "file archived")
    orphan = ThirdParty("data/gone.jsonl", "2" * 64, "MIT")
    result = _scan({"a.md": b"zorblax"}, Policy(STRICT, exceptions=(stale, unused), third_party=(orphan,)))
    assert _terms(result) == [("a.md", "zorblax")]
    assert [entry["path"] for entry in result.exceptions_unused] == ["a.md", "gone.md"]
    assert result.third_party["unused"] == ["data/gone.jsonl"]


@pytest.mark.parametrize(("user", "hit"), [
    ("alice", True), ("user", False), ("User", False), ("me", False), ("runner", False), ("runner2", True)])
def test_home_directories_are_hits_unless_they_name_a_placeholder(user, hit):
    result = _scan({"a.md": f"see /home/{user}/x and /Users/{user}/y".encode()})
    assert _terms(result) == ([("a.md", "path:/Users"), ("a.md", "path:/home")] if hit else [])


@pytest.mark.parametrize(("domain", "hit"), [
    ("corp.io", True), ("example.com", False), ("mail.example.org", False), ("example.net", False),
    ("host.test", False), ("box.invalid", False), ("site.example", False), ("notexample.com", True)])
def test_email_addresses_outside_the_example_domains_are_hits(domain, hit):
    result = _scan({"a.md": ("contact: someone" + "@" + domain + ".").encode()})
    assert _terms(result) == ([("a.md", "email")] if hit else [])


SECRETS = {
    "openai": ("secret:openai-style-key", "sk-" + "proj-" + "a1B2" * 8),
    "google": ("secret:google-api-key", "AI" + "za" + "x" * 35),
    "huggingface": ("secret:huggingface-token", "hf" + "_" + "a" * 34),
    "aws-id": ("secret:aws-access-key-id", "AK" + "IA" + "Q" * 16),
    "github": ("secret:github-token", "gh" + "p_" + "b" * 36),
    "runpod": ("secret:runpod-api-key", "rpa" + "_" + "A1B2C3D4E5" * 4),
    "modal-id": ("secret:modal-token", "ak" + "-" + "a1B2c3D4e5F6g7H8i9J0kL"),
    "modal-secret": ("secret:modal-token", "as" + "-" + "Zy9Xw8Vu7Ts6Rq5Po4Nm3L"),
    "wandb": ("secret:credential-assignment", "WANDB" + "_API_KEY=" + "0123456789abcdef" * 2 + "01234567"),
    "aws-secret": ("secret:credential-assignment", "AWS_SECRET" + "_ACCESS_KEY: " + "wJalrXUtnFEMI/K7MD" + "ENG/bPxRfiCY9"),
    "pem": ("secret:private-key", "-----BEGIN " + "OPENSSH PRIVATE KEY-----"),
    "pgp": ("secret:private-key", "-----BEGIN " + "PGP PRIVATE KEY BLOCK-----"),
}


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_secret_shaped_strings_are_hits_in_every_file_and_are_never_echoed(name):
    kind, secret = SECRETS[name]
    data = f"token = '{secret}'\n".encode()
    policy = Policy(STRICT, third_party=(ThirdParty("d.txt", _sha(data), "MIT"),))
    for result in (_scan({"a.py": data}), _scan({"d.txt": data}, policy)):
        (hit,) = result.hits
        assert hit.term == kind and hit.match is None and hit.context is None
        assert secret not in hit.render() and secret not in str(result.to_json())


def _leaked(secret: str, output: str, width: int = 8) -> bool:
    return any(secret[start:start + width] in output for start in range(len(secret) - width + 1))


def _recorded(result) -> str:
    return "\n".join(hit.render() for hit in result.hits) + json.dumps(result.to_json())


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_no_part_of_a_secret_is_recorded_by_the_hits_around_it(name):
    kind, secret = SECRETS[name]
    home = "/home/" + "alice"
    email = "alice" + "@" + "corp.io"
    result = _scan({"a.py": f"zorblax={secret}  # {home} {email} frobnitz\n".encode(), "b.py": b"zorblax\n"})
    in_a = [hit for hit in result.hits if hit.path == "a.py"]
    assert {hit.term for hit in in_a} == {kind, "zorblax", "path:/home", "email", "frobnitz"}
    assert not _leaked(secret, _recorded(result))
    assert {(hit.match, hit.context) for hit in in_a} == {(None, None)}
    assert [hit.match for hit in result.hits if hit.path == "b.py"] == ["zorblax"]


def test_a_hit_that_overlaps_a_secret_records_nothing_of_it():
    in_url = "gh" + "p_" + "c" * 36
    around_term = "gh" + "p_" + "b" * 12 + "zorblax" + "b" * 17
    text = f"git clone https://{in_url}" + "@" + f"github.com/org/repo.git\nexport T={around_term}\n"
    result = _scan({"a.sh": text.encode()})
    assert sorted(hit.term for hit in result.hits) == [
        "email", "secret:github-token", "secret:github-token", "zorblax"]
    assert not _leaked(in_url, _recorded(result)) and not _leaked(around_term, _recorded(result))


@pytest.mark.parametrize("with_end", [True, False])
def test_a_private_key_block_is_masked_whole_with_or_without_its_end_line(with_end):
    body = "MIIEvQ" + "Ab3x" * 16
    lines = ["-----BEGIN " + "RSA PRIVATE KEY-----", body, body] + (["-----END " + "RSA PRIVATE KEY-----"] * with_end)
    text = "\n".join(lines) + "\n# zorblax\n"
    result = _scan({"k.txt": text.encode()})
    assert sorted(hit.term for hit in result.hits) == ["secret:private-key", "zorblax"]
    assert not _leaked(body, _recorded(result)) and "PRIVATE" not in _recorded(result)
    assert not _leaked(body, scan.redact_secrets(text))


def test_a_private_key_serialized_in_json_without_its_end_line_is_masked_whole():
    body = "MIIEvQ" + "Ab3x" * 16
    text = json.dumps({"k": "-----BEGIN " + f"RSA PRIVATE KEY-----\n{body}\n{body}", "note": "zorblax"})
    result = _scan({"k.json": text.encode()})
    assert sorted(hit.term for hit in result.hits) == ["secret:private-key", "zorblax"]
    assert not _leaked(body, _recorded(result)) and not _leaked(body, scan.redact_secrets(text))


def test_a_secret_one_view_reads_whole_is_not_recorded_split_by_another_view():
    # In the raw bytes a line continuation splits the token, so only the decoded string matches the rule whole.
    token = "gh" + "p_" + "b" * 20 + "c" * 16
    data = _pdf(b"/Title (zorblax " + token[:24].encode() + b"\\\n" + token[24:].encode() + b")", b"q Q")
    result = _scan({"f.pdf": data})
    assert {"secret:github-token", "zorblax"} <= {hit.term for hit in result.hits}
    assert {"text", "latin-1"} & {hit.view for hit in result.hits if hit.term == "zorblax"}
    assert not _leaked(token, _recorded(result))


@pytest.mark.parametrize("name", [*scan.CREDENTIAL_VARIABLES, "WANDB_API_KEY", "RUNPOD_API_KEY", "OPENAI_API_KEY",
                                  "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "MY_SERVER_API_KEY", "API_KEY"])
def test_a_long_value_assigned_to_a_credential_variable_is_a_hit(name):
    value = "Zx9" * 10
    for text in (f"export {name}={value}", f"{name.lower()} = '{value}'", f'{{"{name}": "{value}"}}',
                 f'os.environ["{name}"] = "{value}"'):
        assert _terms(_scan({"a.py": text.encode()})) == [("a.py", "secret:credential-assignment")]


@pytest.mark.parametrize("dotted", ["\u0130", "\u0131"])
def test_a_credential_variable_named_with_a_dotted_or_dotless_i_passes_the_probe_too(dotted):
    value = "Zx9" * 10
    for name in ("WANDB" + "_API_KEY", "MODAL_TOKEN" + "_ID"):
        text = f"{name.replace('I', dotted)}={value}"
        assert _terms(_scan({"a.py": text.encode()})) == [("a.py", "secret:credential-assignment")]
        assert value not in scan.redact_secrets(text)


def test_a_secret_in_a_name_is_a_hit_and_masked_wherever_the_name_is_recorded():
    token = "hf" + "_" + "d" * 34
    scanner = Scanner(POLICY)
    scanner.scan_file(f"keys/{token}/.env", b"zorblax\n")
    scanner.scan_file(f"zorblax-{token}.bin", b"\x1f\x8b broken")
    scanner.flag(f"links/{token}", "symlink")
    result = scanner.result()
    shown = scan.SECRET_MASK * len(token)
    assert sorted((hit.path, hit.term, hit.view) for hit in result.hits) == [
        (f"keys/{shown}/.env", "forbidden-name:.env*", "name"),
        (f"keys/{shown}/.env", "secret:huggingface-token", "name"),
        (f"keys/{shown}/.env", "zorblax", "text"),
        (f"links/{shown}", "symlink", "name"),
        (f"zorblax-{shown}.bin", "secret:huggingface-token", "name"),
        (f"zorblax-{shown}.bin", "unscannable", "gunzip"),
        (f"zorblax-{shown}.bin", "zorblax", "name"),
    ]
    assert not _leaked(token, _recorded(result))


def test_redact_secrets_masks_each_character_and_keeps_the_rest():
    key = "AK" + "IA" + "Q" * 16
    assert scan.redact_secrets(f"id={key};") == "id=" + scan.SECRET_MASK * len(key) + ";"
    assert scan.redact_secrets("nothing here") == "nothing here"


def test_a_github_fine_grained_token_is_a_hit():
    result = _scan({"a.py": ("github" + "_pat_" + "c" * 30).encode()})
    assert _terms(result) == [("a.py", "secret:github-token")]


@pytest.mark.parametrize("text", [
    "sk-" + "ExampleKeyPlaceholderValue",  # no digit: a placeholder, not a key
    "task-" + "a1" * 12,  # 'sk-' inside a word
    "hf_datasets_cache",
    "AK" + "IA" + "Q" * 17,  # too long for a key id
    "has-" + "a1" * 12,  # 'as-' inside a word
    "as-" + "a1" * 12 + "-slug",  # a hyphenated slug, not a token
    "WANDB" + "_API_KEY=your-wandb-api-key-here",  # a placeholder: no digit
    "WANDB" + "_API_KEY=$WANDB_API_KEY_FROM_THE_ENVIRONMENT_1",
    "WANDB" + "_API_KEY_FILE=/run/secrets/wandb-key-2026-file",
    "api_key" + "_env: OPENROUTER_API_KEY_2026_ROTATED_VARIABLE",  # names the variable, not the key
])
def test_key_lookalikes_are_not_hits(text):
    assert _scan({"a.py": text.encode()}).ok


DASHES = [chr(code) for code, ascii_code in sorted(scan.LOOKALIKES.items()) if ascii_code == "-"]
OTHER_LOOKALIKES = [chr(code) for code, ascii_code in sorted(scan.LOOKALIKES.items()) if ascii_code != "-"]


@pytest.mark.parametrize("dash", DASHES)
def test_a_unicode_dash_that_touches_an_address_or_a_token_does_not_hide_it(dash):
    email = "jane.doe" + "@" + "corp-lab.org"
    assert _terms(_scan({"a.md": f"Questions: {email}{dash}or open an issue".encode()})) == [("a.md", "email")]
    for prefix in ("ak-", "as-"):
        token = prefix + "a1B2c3D4e5F6g7H8i9J0kL"
        for text in (f"id{dash}{token}", f"{token}{dash}next", f"{dash}{token}{dash}"):
            assert _terms(_scan({"a.md": text.encode()})) == [("a.md", "secret:modal-token")]
            assert not _leaked(token, scan.redact_secrets(text))


@pytest.mark.parametrize("quotes", ["\u201c\u201d", "\u201e\u201c", "\u201f\u201d", "\u2018\u2019", "\u201a\u2019"])
def test_a_credential_assigned_in_curly_quotes_is_a_hit_and_masked(quotes):
    # Word processors write smart quotes: the fold reads them as the ASCII quote the rule expects.
    value = "Q1w2E3r4" * 3
    open_, close = quotes
    for text in (f"WANDB_API_KEY = {open_}{value}{close}", f"{open_}WANDB_API_KEY{close}: {open_}{value}{close}"):
        assert _terms(_scan({"a.md": text.encode()})) == [("a.md", "secret:credential-assignment")]
        assert not _leaked(value, scan.redact_secrets(text))


FOLD_SAMPLES = (
    "zorblax", "Q-Lab-77", "4815162342", "frobnitz", "wug corp", "/home/" + "alice", "/home/" + "user",
    "/Users/" + "me", "alice" + "@" + "corp-lab.io", *(secret for _, secret in SECRETS.values()),
)


def _found(texts: list[str]) -> list[tuple[set[tuple], set[int]]]:
    found = []
    for text in texts:
        scanner = Scanner(POLICY)
        scanner.scan_file("a.md", text.encode())
        scanner.scan_file(text, b"")
        hits = {(hit.path == "a.md", hit.term, hit.kind, hit.view, hit.offset) for hit in scanner.result().hits}
        found.append((hits, {at for at, shown in enumerate(scan.redact_secrets(text)) if shown == scan.SECRET_MASK}))
    return found


@pytest.mark.parametrize("char", DASHES + OTHER_LOOKALIKES)
def test_the_fold_only_adds_hits_and_masked_characters(char, monkeypatch):
    texts = [text for sample in FOLD_SAMPLES
             for text in (f"{char}{sample}{char}", f"x{char}{sample}", f"{sample}{char}x", f"{sample}{char}{sample}")]
    with_fold = _found(texts)
    monkeypatch.setattr(scan, "LOOKALIKES", {})
    without_fold = _found(texts)
    for text, (hits, masked), (hits_as_written, masked_as_written) in zip(texts, with_fold, without_fold):
        assert hits_as_written <= hits and masked_as_written <= masked, repr(text)
    # The samples stand for every term and every rule but the /Users one, each a hit as written.
    assert {hit[1] for hits, _ in without_fold for hit in hits} >= {
        *STRICT, *BROAD, "email", "path:/home", *(kind for kind, _ in SECRETS.values())}


def test_a_gzip_member_is_decompressed_and_scanned():
    data = gzip.compress(b"hello\nzorblax\n", mtime=0)
    (hit,) = _scan({"cache/x.jsonl.gz": data}).hits
    assert (hit.view, hit.line, hit.term) == ("gunzip/text", 2, "zorblax")


def test_gzip_that_cannot_be_decompressed_is_unscannable():
    nested = b"x"
    for _ in range(scan.MAX_DEPTH + 1):
        nested = gzip.compress(nested, mtime=0)
    corrupt = scan.GZIP_MAGIC + b"\x08\x00not a gzip body"
    result = _scan({"deep.gz": nested, "bad.gz": corrupt})
    assert _terms(result) == [("bad.gz", "unscannable"), ("deep.gz", "unscannable")]
    assert {hit.view for hit in result.hits} == {"gunzip", "gunzip/" * scan.MAX_DEPTH + "gunzip"}


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _tar_bytes(members: dict[str, bytes], mode: str = "w") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize("data", [b"PK\x03\x04\x14\x00", b"\xfd7zXZ\x00\x00", b"7z\xbc\xaf\x27\x1c\x00",
                                  b"BZh91AY&SY"])
def test_an_archive_the_scan_cannot_open_is_unscannable(data):
    (hit,) = _scan({"bundle.bin": data}).hits
    assert hit.term == "unscannable" and hit.view.startswith("archive:")


LEAKY = b"by zorblax\n" * 50


@pytest.mark.parametrize(("data", "view"), [
    (b"\n" + _zip_bytes({"notes.txt": LEAKY}), "archive:zip"),  # a zip still opens after a prefix
    (_tar_bytes({"x.jsonl.gz": gzip.compress(LEAKY, mtime=0)}), "archive:tar"),
    (_tar_bytes({"bundle.zip": _zip_bytes({"notes.txt": LEAKY})}, "w:gz"), "gunzip/archive:tar"),
], ids=["zip-after-prefix", "gzip-in-tar", "zip-in-gzipped-tar"])
def test_an_archive_after_a_prefix_or_inside_a_tar_is_unscannable(data, view):
    result = _scan({"a.bin": data})
    assert (not result.ok) and ("a.bin", "unscannable", view) in {(hit.path, hit.term, hit.view) for hit in result.hits}


def test_gzip_larger_than_the_inflate_limit_is_unscannable(monkeypatch):
    monkeypatch.setattr(scan, "MAX_INFLATE", 16)
    assert _terms(_scan({"big.gz": gzip.compress(b"a" * 64, mtime=0)})) == [("big.gz", "unscannable")]


def _pdf(info: bytes, stream: bytes, *, flate: bool = True) -> bytes:
    body = zlib.compress(stream) if flate else stream
    return (b"%PDF-1.4\n1 0 obj\n<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body
            + b"\nendstream\nendobj\n2 0 obj\n<< " + info + b" >>\nendobj\ntrailer\n<< /Info 2 0 R >>\n%%EOF\n")


def _utf16_hex(text: str) -> bytes:
    return b"<FEFF" + text.encode("utf-16-be").hex().upper().encode() + b">"


def _utf16_octal(text: str) -> bytes:
    return b"(\\376\\377" + b"".join(b"\\%03o" % byte for byte in text.encode("utf-16-be")) + b")"


@pytest.mark.parametrize(("info", "raw_visible"), [
    (b"/Author (Jane Zorblax)", True),
    (b"/Author " + _utf16_hex("Jane Zorblax"), False),
    (b"/Author " + _utf16_octal("Jane Zorblax"), False),
    (b"/Author (\xef\xbb\xbfJane Zor\\\nblax)", False),
    (b"/Author <4A616E65205A6F72626C6178 >", False),
    (b"/Author (Jane \\(Z\\)\\t Zo\\162blax)", False),
])
def test_a_pdf_author_is_found_however_the_string_is_encoded(info, raw_visible):
    hits = _scan({"figures/f.pdf": _pdf(info, b"q 1 0 0 1 0 0 cm Q")}).hits
    assert {hit.term for hit in hits} == {"zorblax"}
    assert "pdf-strings" in {hit.view for hit in hits}
    assert bool({"text", "latin-1"} & {hit.view for hit in hits}) is raw_visible


def test_a_pdf_flate_stream_is_decompressed_and_its_strings_decoded():
    stream = b"BT /F1 12 Tf (drawn text) Tj " + _utf16_hex("zorblax") + b" Tj ET"
    views = {hit.view for hit in _scan({"f.pdf": _pdf(b"/Producer (x)", stream)}).hits}
    assert any(view.startswith("pdf-stream@") and view.endswith("/strings") for view in views)


def test_pdf_xmp_metadata_is_read_compressed_or_not():
    xmp = (b'<?xpacket begin=""?><x:xmpmeta xmlns:x="adobe:ns:meta/"><dc:creator><rdf:Seq>'
           b'<rdf:li>Zorblax</rdf:li></rdf:Seq></dc:creator></x:xmpmeta><?xpacket end="r"?>')
    for flate in (True, False):
        hits = _scan({"f.pdf": _pdf(b"/Metadata 1 0 R", xmp, flate=flate)}).hits
        assert {hit.term for hit in hits} == {"zorblax"}
        assert any(hit.view.startswith("pdf-stream@") for hit in hits) is flate


def test_a_pdf_stream_that_is_not_flate_is_read_raw():
    data = _pdf(b"/Producer (x)", b"BT (zorblax) Tj ET", flate=False)
    result = _scan({"f.pdf": data})
    assert {hit.view for hit in result.hits} == {"text", "pdf-strings"}
    assert "pdf-stream" not in result.views


def test_a_kerned_tj_array_is_read_joined():
    stream = b"BT [ (F) 12.5 (r) -7 <6F626E69747A> ] TJ [(Zor)-40(blax)]TJ ET"
    hits = _scan({"f.pdf": _pdf(b"/Producer (x)", stream)}).hits
    assert {(hit.term, hit.match) for hit in hits} == {("frobnitz", "Frobnitz"), ("zorblax", "Zorblax")}


@pytest.mark.parametrize(("stream", "found"), [
    # One glyph per Tj, positioned by Td between them, as matplotlib's mathtext draws.
    (b"BT /F1 11 Tf (Zor) Tj 7 0 Td (b) Tj 3 0 Td <6C6178> Tj ET", True),
    # A name spelled like the operator does not end the text object.
    (b"BT (Zor) Tj /ET gs (blax) Tj ET", True),
    # ET does: a string shown after it is not joined to the text before it.
    (b"BT (Zor) Tj ET (blax) Tj", False),
])
def test_the_strings_of_one_text_object_are_read_joined(stream, found):
    hits = _scan({"f.pdf": _pdf(b"/Producer (x)", stream)}).hits
    assert {hit.term for hit in hits} == ({"zorblax"} if found else set())


def test_a_bomless_utf16_string_is_read_as_matplotlib_writes_type42_text():
    title = b"(" + "Run by Zorblax".encode("utf-16-be").replace(b"(", b"\\(") + b")"
    kerned = b"[ (" + "Frob".encode("utf-16-be") + b") 62.5 (" + "nitz".encode("utf-16-be") + b") ] TJ"
    hits = _scan({"f.pdf": _pdf(b"/Producer (x)", b"BT " + title + b" Tj " + kerned + b" ET")}).hits
    assert {(hit.term, hit.match) for hit in hits} == {("frobnitz", "Frobnitz"), ("zorblax", "Zorblax")}


def test_a_literal_string_may_hold_balanced_parentheses():
    (hit,) = [hit for hit in _scan({"f.pdf": _pdf(b"/Title (Jane (Z\\) Zo\\162blax)", b"")}).hits
              if hit.view == "pdf-strings"]
    assert hit.match == "Zorblax"


def test_an_odd_length_hex_string_is_padded_as_pdf_specifies():
    # 15 hex digits: "zorblax" and a lone '7', which pads with a trailing 0 to 0x70 ('p').
    hits = _scan({"f.pdf": _pdf(b"/Title <7A6F72626C6178 7>", b"")}).hits
    assert [hit.view for hit in hits] == ["pdf-strings"]


@pytest.mark.parametrize("fonttype", [42, 3])
def test_a_matplotlib_figure_with_identifying_text_is_caught(fonttype, tmp_path):
    # From the real producer, under the repository's figure style (Type 42 text is UTF-16 without a mark).
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ruleforge.experiments.style import SOFT_STYLE

    with matplotlib.rc_context({**SOFT_STYLE, "pdf.fonttype": fonttype}):
        figure, axes = plt.subplots()
        axes.set_title("Run by Zorblax on Frobnitz")
        axes.set_xticks([0, 1], ["AVAWAY Tolerance", "wug corp"])
        figure.savefig(tmp_path / "f.pdf", metadata={"Author": "Q-Lab-77"})
        figure.savefig(tmp_path / "f.png", metadata={"Author": "Q-Lab-77"})
        plt.close(figure)
    result = _scan({"f.pdf": (tmp_path / "f.pdf").read_bytes(), "f.png": (tmp_path / "f.png").read_bytes()})
    assert sorted({(hit.path, hit.term) for hit in result.hits}) == [
        ("f.pdf", "frobnitz"), ("f.pdf", "q-lab-77"), ("f.pdf", "wug corp"), ("f.pdf", "zorblax"),
        ("f.png", "q-lab-77")]


def _label_pdf(label: str, fonttype: int, tmp_path, **style: str) -> bytes:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ruleforge.experiments.style import SOFT_STYLE

    with matplotlib.rc_context({**SOFT_STYLE, "pdf.fonttype": fonttype, **style}):
        figure, axes = plt.subplots()
        axes.set_xlabel(label)
        figure.savefig(tmp_path / "f.pdf")
        plt.close(figure)
    return (tmp_path / "f.pdf").read_bytes()


@pytest.mark.parametrize("fonttype", [42, 3])
@pytest.mark.parametrize(("label", "terms"), [
    (r"$\mathrm{Zorblax}$", {"zorblax"}),
    (r"Run by $\it{Zorblax}$ on Frobnitz", {"zorblax", "frobnitz"}),
    (r"$\mathrm{Frobnitz}$ wug corp", {"frobnitz", "wug corp"}),
    # Each line is its own text object: consecutive ones are adjacent lines of the same reading.
    ("by the wug\ncorp team", {"wug corp"}),
    (r"$\mathrm{Frobnitz\ Zorblax}$", {"zorblax"}),
    (r"$\mathrm{q-lab-77}$", {42: {"q-lab-77"}, 3: set()}),
    (r"by $\mathrm{Q-Lab-77}$", {42: {"q-lab-77"}, 3: set()}),
    (r"$\mathrm{wug-corp}$", {42: {"wug corp"}, 3: set()}),
    (r"$\mathrm{q}$-lab-77", {"q-lab-77"}),  # a hyphen outside math mode is a glyph of its own
])
def test_matplotlib_text_drawn_in_pieces_is_read_as_words(label, terms, fonttype, tmp_path):
    expected = terms[fonttype] if isinstance(terms, dict) else terms
    assert {hit.term for hit in _scan({"f.pdf": _label_pdf(label, fonttype, tmp_path)}).hits} == expected


@pytest.mark.parametrize("fonttype", [42, 3])
def test_mathtext_symbols_drawn_from_computer_modern_read_as_that_fonts_codes(fonttype, tmp_path):
    policy = Policy(("q'lab", "q-lab-77"))

    def read(pdf: bytes) -> str:
        return "\n".join(text for _, text in scan.views(pdf) if text)

    prime = _label_pdf(r"$\mathrm{q'lab}$", fonttype, tmp_path)
    assert "q0lab" in read(prime) and _scan({"f.pdf": prime}, policy).ok
    minus = _label_pdf(r"$\mathrm{q-lab-77}$", fonttype, tmp_path, **{"mathtext.fontset": "cm"})
    assert "q\u00a1lab\u00a177" in read(minus) and _scan({"f.pdf": minus}, policy).ok
    plain = _label_pdf("q'lab and q-lab-77", fonttype, tmp_path)
    assert set(_terms(_scan({"f.pdf": plain}, policy))) == {("f.pdf", "q'lab"), ("f.pdf", "q-lab-77")}


def test_the_prime_a_stix_font_writes_in_type42_text_reads_as_an_apostrophe(tmp_path):
    # Where a file does carry U+2032 PRIME (mathtext's STIX fonts in Type 42 text, or pasted text), it folds.
    pdf = _label_pdf(r"$\mathrm{q'lab}$", 42, tmp_path, **{"mathtext.fontset": "stix"})
    assert set(_terms(_scan({"f.pdf": pdf}, Policy(("q'lab",))))) == {("f.pdf", "q'lab")}


def test_non_ascii_text_in_a_view_read_as_latin1_is_not_folded():
    for leaky, found in (("Q\u2011Lab\u201177", False), ("Q-Lab-77", True)):
        cases = {"blob.bin": b"\x00by " + leaky.encode(),
                 "stream.pdf": _pdf(b"/Producer (x)", b"<dc:creator>" + leaky.encode() + b"</dc:creator>")}
        assert {path for path, _ in _terms(_scan(cases))} == ({"blob.bin", "stream.pdf"} if found else set())
    assert _scan({"f.pdf": _pdf(b"/Author (Q\\212Lab\\21277)", b"q Q")}).ok
    assert set(_terms(_scan({"f.pdf": _pdf(b"/Author (Q-Lab-77)", b"q Q")}))) == {("f.pdf", "q-lab-77")}


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _png(*chunks: bytes, end: bool = True) -> bytes:
    header = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    return scan.PNG_SIGNATURE + header + b"".join(chunks) + (_chunk(b"IEND", b"") if end else b"")


@pytest.mark.parametrize(("chunk", "view"), [
    (_chunk(b"tEXt", b"Author\0Zorblax"), "png:tEXt"),
    (_chunk(b"zTXt", b"Comment\0\0" + zlib.compress(b"by zorblax")), "png:zTXt"),
    (_chunk(b"iTXt", b"Author\0\0\0en\0Autor\0zorblax"), "png:iTXt"),
    (_chunk(b"iTXt", b"Author\0\x01\0en\0\0" + zlib.compress(b"zorblax")), "png:iTXt"),
])
def test_png_text_chunks_are_read(chunk, view):
    hits = _scan({"f.png": _png(chunk)}).hits
    assert view in {hit.view for hit in hits if hit.term == "zorblax"}


def test_a_png_text_chunk_that_does_not_inflate_is_unscannable_and_chunks_after_iend_are_ignored():
    broken = _chunk(b"zTXt", b"Comment\0\0" + zlib.compress(b"zorblax")[:-3])
    after_end = _png(end=True) + _chunk(b"tEXt", b"k\0zorblax")
    truncated = _png(_chunk(b"tEXt", b"k\0fine"), end=False) + b"\x00\x00"
    result = _scan({"a.png": _png(broken), "b.png": after_end, "c.png": truncated})
    assert [(hit.path, hit.term, hit.view) for hit in result.hits if hit.view.startswith("png")] == [
        ("a.png", "unscannable", "png:zTXt")]


@pytest.mark.parametrize(("path", "reason"), [
    (".env", ".env*"), ("config/.env.local", ".env*"), (".envrc", ".env*"), (".env_local", ".env*"),
    (".ENV-prod", ".env*"), (".envs/x.txt", ".env*"), (".env.example", None), ("a/.env.example", None),
    (".netrc", ".netrc"), ("_netrc", "_netrc"), (".git-credentials", ".git-credentials"), (".pypirc", ".pypirc"),
    ("keys/id_ed25519", "id_ed25519*"), ("id_ecdsa.pub", "id_ecdsa*"), ("id_dsa", "id_dsa*"),
    ("keys/server.pem", "*.pem"), ("id_rsa.pub", "id_rsa*"), ("pkg/__pycache__/m.cpython-312.pyc", "__pycache__"),
    ("m.pyc", "*.pyc"), (".git/config", ".git"), ("sub/.git", ".git"), ("src/pkg.egg-info/PKG-INFO", "*.egg-info"),
    ("CLAUDE.md", "claude.md"), ("docs/Claude.md", "claude.md"), (".gitignore", None),
    (".claude/settings.json", ".claude"), (".coverage", ".coverage"), (".coverage.host.1", ".coverage.*"),
    (".DS_Store", ".ds_store"), ("README.md", None), ("experiments/.gitkeep", None), (".gitattributes", None),
    ("runpod.env", "*.env"), ("configs/modal.env", "*.env"), ("PROD.ENV", "*.env"),
])
def test_forbidden_names(path, reason):
    assert forbidden_reason(path) == reason


@pytest.mark.parametrize(("name", "unsafe"), [
    ("code/a.py", False), ("code/dir/", False), ("../a", True), ("/abs", True), ("a//b", True),
    ("a/./b", True), ("a\\b", True), ("a\0b", True),
])
def test_unsafe_names(name, unsafe):
    assert unsafe_name(name) is unsafe


def test_a_member_name_is_scanned_for_both_term_lists_even_in_third_party_data():
    data = b"{}\n"
    policy = Policy(STRICT, BROAD, third_party=(ThirdParty("data/annotations_Frobnitz.jsonl", _sha(data), "MIT"),))
    result = _scan({"data/annotations_Frobnitz.jsonl": data, ".env": b"", "../up": b""}, policy)
    assert [(hit.path, hit.term, hit.kind, hit.view) for hit in result.hits] == [
        ("data/annotations_Frobnitz.jsonl", "frobnitz", "broad", "name"),
        (".env", "forbidden-name:.env*", "name", "name"),
        ("../up", "unsafe-name", "name", "name"),
    ]


def test_a_directory_scan_flags_links_special_files_and_forbidden_directories_once(tmp_path):
    root = tmp_path / "code"
    (root / "src").mkdir(parents=True)
    (root / "src" / "ok.py").write_text("x = 1\n")
    (root / "src" / "leak.md").write_text("by zorblax\n")
    (root / ".git" / "objects").mkdir(parents=True)
    (root / ".git" / "objects" / "a").write_text("zorblax")
    (root / ".git" / "HEAD").write_text("ref")
    (root / "pkg" / "__pycache__").mkdir(parents=True)
    (root / ".env" / "bin").mkdir(parents=True)  # a virtual environment named .env
    (root / ".env" / "bin" / "activate").write_text("zorblax")
    (root / "link.md").symlink_to(root / "src" / "ok.py")
    (root / "linkdir").symlink_to(root / "src")
    os.mkfifo(root / "pipe")
    result = scan_dir(root, POLICY)
    assert sorted((hit.path, hit.term) for hit in result.hits) == [
        (".env", "forbidden-name:.env*"), (".git", "forbidden-name:.git"), ("link.md", "symlink"),
        ("linkdir", "symlink"),
        ("pipe", "special-file"), ("pkg/__pycache__", "forbidden-name:__pycache__"), ("src/leak.md", "zorblax")]
    assert result.files == 2


def test_a_directory_scan_refuses_a_path_that_is_not_a_directory(tmp_path):
    with pytest.raises(ReleaseError, match="not a directory"):
        scan_dir(tmp_path / "missing", POLICY)


def test_a_directory_scan_refuses_an_empty_tree_unless_allowed(tmp_path):
    with pytest.raises(ReleaseError, match="no file to scan"):
        scan_dir(tmp_path, POLICY)
    assert scan_dir(tmp_path, POLICY, allow_empty=True).files == 0
    (tmp_path / "link").symlink_to(tmp_path)
    assert _terms(scan_dir(tmp_path, POLICY)) == [("link", "symlink")]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-0 path")
@pytest.mark.parametrize("locked", ["sub", "sub/leak.md", "."])
def test_a_directory_scan_refuses_what_it_cannot_read(tmp_path, locked):
    root = tmp_path / "code"
    (root / "sub").mkdir(parents=True)
    (root / "ok.md").write_text("fine\n")
    (root / "sub" / "leak.md").write_text("zorblax\n")
    target = root / locked
    target.chmod(0)
    try:
        with pytest.raises(ReleaseError, match="cannot read .*Permission denied"):
            scan_dir(root, POLICY)
    finally:
        target.chmod(0o700)
    assert _terms(scan_dir(root, POLICY)) == [("sub/leak.md", "zorblax")]


def test_hits_render_with_their_location_and_the_result_serializes(tmp_path):
    assert Hit("a.py", "zorblax", "strict", "text", 7, 2, "Zorblax").render() == \
        "a.py:2 [text@7] strict:zorblax 'Zorblax'"
    assert Hit("link", "symlink", "name", "name").render() == "link [name@0] name:symlink"
    result = _scan({"a.md": b"zorblax zorblax frobnitz", "b.pdf": _pdf(b"/Producer (x)", b"q Q")})
    assert result.summary() == {"a.md": {"frobnitz": 1, "zorblax": 2}}
    payload = result.to_json()
    assert payload["hits"][0]["path"] == "a.md" and payload["excepted_hits"] == []
    assert {view: payload["views"][view] for view in ("pdf-stream", "pdf-stream/strings", "pdf-strings")} == {
        "pdf-stream": 1, "pdf-stream/strings": 1, "pdf-strings": 1}
    assert result.statistics()["hits"] == 3


ADDRESS = "author" + "@" + "lab.edu"


def _tree(tmp_path, files):
    for relative, data in files.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(data)
    return tmp_path


def test_a_public_identity_is_allowed_only_in_the_files_it_is_listed_for(tmp_path):
    root = _tree(tmp_path, {"pyproject.toml": f"email = '{ADDRESS}'".encode(),
                            "src/a.py": f"# {ADDRESS.upper()}".encode()})
    result = scan_tree(root, ["pyproject.toml", "src/a.py"], TreePolicy(identities={ADDRESS: ("pyproject.toml",)}))
    assert [(hit.path, hit.term) for hit in result.hits] == [("src/a.py", "email")]
    assert [hit.path for hit in result.allowed] == ["pyproject.toml"] and result.unused == [] and not result.ok


def test_an_allowance_excuses_its_terms_in_its_files_and_never_a_secret(tmp_path):
    secret = "hf" + "_" + "a" * 34
    root = _tree(tmp_path, {"data/a.jsonl": f"see {'/home/' + 'alice'}/x, {ADDRESS}".encode(),
                            "data/b.jsonl": f"token {secret}".encode()})
    policy = TreePolicy(allowances=(TreeAllowance("data/*.jsonl", frozenset({"email", "path:/home",
                                                                            "secret:huggingface-token"}), "upstream"),))
    result = scan_tree(root, ["data/a.jsonl", "data/b.jsonl"], policy)
    assert [(hit.path, hit.term) for hit in result.hits] == [("data/b.jsonl", "secret:huggingface-token")]
    assert sorted(hit.term for hit in result.allowed) == ["email", "path:/home"]


def test_an_entry_nothing_matched_is_reported_as_stale(tmp_path):
    root = _tree(tmp_path, {"a.md": b"clean"})
    policy = TreePolicy(identities={ADDRESS: ("a.md",)}, allowances=(TreeAllowance("data/*", frozenset({"email"}), "x"),))
    result = scan_tree(root, ["a.md"], policy)
    assert result.hits == [] and result.unused == [f"identity {ADDRESS}", "allowance data/*"] and not result.ok
    assert scan_tree(root, ["a.md"], TreePolicy()).ok


def test_a_tree_path_that_is_not_a_regular_file_is_refused(tmp_path):
    root = _tree(tmp_path, {"a.md": b"x"})
    (root / "link.md").symlink_to(root / "a.md")
    for relative in ("link.md", "missing.md"):
        with pytest.raises(ReleaseError, match="not a regular file"):
            scan_tree(root, [relative], TreePolicy())


REPOSITORY = Path(__file__).resolve().parents[1]
#: The site the paper's jobs ran on, which no public file names: its cluster partition (and its nodes' domain), the
#: cluster's name and its login host's domain, its node hosts and the account the jobs ran under, each assembled here
#: so this file holds none of them.
SITE_NAMES = ("volt" + "agepark", "schmidt" + "sciences", "parallel" + ".works", "peter" + "henderson",
              *(f"g0{number}" for number in range(374, 382)))
PUBLIC = TreePolicy(
    identities={"michel.liao22" + "@" + "gmail.com": ("pyproject.toml",)},
    allowances=(
        TreeAllowance("data/populations/*.jsonl", frozenset({"email"}),
                      "third-party scenario text, as published upstream (data/README.md)"),
        TreeAllowance("data/scenarios/*/modelspec/*.jsonl", frozenset({"email", "path:/home"}),
                      "third-party scenario text, as published upstream (data/README.md)"),
        TreeAllowance("data/rules/modelspec/openai_rules_all_full.jsonl", frozenset({"email"}),
                      "the OpenAI Model Spec's text"),
        TreeAllowance("tests/test_launch_guards.py", frozenset({"email"}),
                      "Hydra package overrides (group@package), which read as addresses"),
        TreeAllowance("tests/identity/fixtures/settlement.jsonl.gz", frozenset({"email"}),
                      "compressed bytes that read as an address by chance; held to COMPRESSED_HITS"),
    ),
    forbidden=SITE_NAMES,
)


def repository_files(root: Path) -> list[str]:
    """The files a commit would hold; without git, every file under ``root`` but dot-directories and caches."""
    if (root / ".git").exists():
        listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
                                capture_output=True, check=True).stdout.decode("utf-8").split("\0")
        return sorted(path for path in listed if path and (root / path).is_file())
    found = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")
                             and name not in ("__pycache__", "experiments") and not name.endswith(".egg-info")]
        found += [(Path(directory) / name).relative_to(root).as_posix() for name in files]
    return sorted(found)


#: The one compressed file whose bytes read as an address by chance, by its SHA-256, and that one hit by view, offset and
#: matched text: any other hit in it, in its gzip header fields or in the rows it decompresses to, is refused.
COMPRESSED_FIXTURE = "tests/identity/fixtures/settlement.jsonl.gz"
COMPRESSED_FIXTURE_SHA256 = "d97be1b7112e1510e6790faa302bac88cec2054738a115708abd8abae06a16ec"
COMPRESSED_HITS = [("latin-1", 756739, "R" + "@" + "6.uP")]


def public_problems(root: Path, files: list[str]) -> list[str]:
    """What the public policy refuses in ``files`` of ``root``, the compressed fixture held to its one known hit."""
    result = scan_tree(root, files, PUBLIC)
    problems = [hit.render() for hit in result.hits] + [f"unused {key}" for key in result.unused]
    allowed = [(hit.view, hit.offset, hit.match) for hit in result.allowed if hit.path == COMPRESSED_FIXTURE]
    if COMPRESSED_FIXTURE in files and (_sha((root / COMPRESSED_FIXTURE).read_bytes()) != COMPRESSED_FIXTURE_SHA256
                                        or allowed != COMPRESSED_HITS):
        problems.append(f"{COMPRESSED_FIXTURE}: not the fixture whose one hit is {COMPRESSED_HITS}: {allowed}")
    return problems


def test_the_repository_holds_no_secret_private_path_or_address():
    files = repository_files(REPOSITORY)
    assert len(files) > 300 and "pyproject.toml" in files and COMPRESSED_FIXTURE in files
    assert public_problems(REPOSITORY, files) == []


@pytest.mark.parametrize("name", SITE_NAMES)
def test_a_file_naming_the_site_is_refused_wherever_it_names_it(tmp_path, name):
    """The site's names, in a document, a test and a config, plain or in another case and joined by a separator."""
    spelled = (name, name.upper(), f"/{name}/", f"{name}.prod")
    root = _tree(tmp_path, {"docs/a.md": f"ran on {spelled[0]}\n".encode(), "tests/b.py": f"X = '{spelled[1]}'\n".encode(),
                            "configs/c.yaml": f"dir: {spelled[2]}{spelled[3]}\n".encode()})
    result = scan_tree(root, ["docs/a.md", "tests/b.py", "configs/c.yaml"], PUBLIC)
    assert sorted({hit.path for hit in result.hits if hit.term == name}) == ["configs/c.yaml", "docs/a.md", "tests/b.py"]
    assert scan_tree(_tree(tmp_path / "clean", {"docs/a.md": b"ran on a cluster\n"}), ["docs/a.md"],
                     TreePolicy(forbidden=SITE_NAMES)).ok


@pytest.mark.parametrize("flag", [0x08, 0x10], ids=["FNAME", "FCOMMENT"])
def test_an_address_in_the_compressed_fixture_s_gzip_header_is_refused(tmp_path, flag):
    original = (REPOSITORY / COMPRESSED_FIXTURE).read_bytes()
    address = "ada" + "@" + "lab.edu"
    planted = original[:3] + bytes([original[3] | flag]) + original[4:10] + f"{address}\0".encode() + original[10:]
    assert original[3] == 0 and gzip.decompress(planted) == gzip.decompress(original)
    root = _tree(tmp_path, {COMPRESSED_FIXTURE: planted})
    problems = public_problems(root, [COMPRESSED_FIXTURE])
    assert any(problem.startswith(f"{COMPRESSED_FIXTURE}:") and address in problem for problem in problems)
    kept = public_problems(_tree(tmp_path / "kept", {COMPRESSED_FIXTURE: original}), [COMPRESSED_FIXTURE])
    assert not [problem for problem in kept if COMPRESSED_FIXTURE in problem or "rule:" in problem]


def test_a_planted_tree_fails_the_repository_s_policy_naming_each_file(tmp_path):
    home = "/home/" + "alice" + "/slurm" + "_scratch/run-7/out.log"
    planted = {"docs/notes.md": f"the logs are under {home}\n".encode(),
               "src/pkg/contact.py": f"OWNER = '{'ada' + '@' + 'lab.edu'}'\n".encode(),
               "configs/keys.yaml": f"token: {'hf' + '_' + 'Qx7' * 12}\n".encode(),
               "README.md": b"nothing to see here\n"}
    root = _tree(tmp_path, planted)
    result = scan_tree(root, sorted(planted), PUBLIC)
    assert sorted((hit.path, hit.term) for hit in result.hits) == [
        ("configs/keys.yaml", "secret:huggingface-token"), ("docs/notes.md", "path:/home"),
        ("src/pkg/contact.py", "email")]
    assert not result.ok and all("Qx7" not in hit.render() for hit in result.hits)
