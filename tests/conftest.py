"""
Shared test setup.

- Network guard: every test runs offline. Any connection to a non-loopback
  address raises, so a test can never quietly depend on UniProt, Hugging Face
  or Docker Hub. Loopback stays allowed (Windows uses it internally).
- A real EsmTokenizer built from ESM-2's vocabulary, written to a temp file,
  so dataset tests need no download.
- Small record factories for synthetic data.
"""

import ipaddress
import socket

import pytest
from transformers import EsmTokenizer


class NetworkBlockedError(RuntimeError):
    pass


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            raise NetworkBlockedError(f"Network access blocked in tests: {address}")
        return real_connect(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        raise NetworkBlockedError(f"Network access blocked in tests: {address}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


# ESM-2 vocabulary, in order (facebook/esm2_* vocab.txt): <cls>=0, <pad>=1, <eos>=2
ESM2_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
    "<null_1>", "<mask>",
]


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory):
    vocab_file = tmp_path_factory.mktemp("esm2") / "vocab.txt"
    vocab_file.write_text("\n".join(ESM2_VOCAB) + "\n")
    return EsmTokenizer(vocab_file=str(vocab_file))


def make_record(uid: str, sequence: str = "MKTAYIAK", ecs=("3.4.21.4",), **extra) -> dict:
    """A record in the download.py schema."""
    ec_numbers = list(ecs)
    record = {
        "uniprot_id": uid,
        "sequence": sequence,
        "ec_numbers": ec_numbers,
        "ec_subclass": ".".join(ec_numbers[0].split(".")[:2]),
        "protein_name": f"protein {uid}",
        "organism": "Homo sapiens",
        "length": len(sequence),
    }
    record.update(extra)
    return record
