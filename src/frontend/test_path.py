"""
test_path.py — Python ctypes binding for the TestPath C API.

Provides a Pythonic interface to inject requests into the pypto-serving
engine and retrieve responses, without any network stack.

Usage:
    from test_path import TestPathClient

    client = TestPathClient("path/to/pypto_testpath.so")
    client.start()

    response = client.infer(
        token_ids=[101, 2023, 2003, 1037],
        max_tokens=4,
        vocab_size=32,
    )
    print(f"Output tokens: {response}")

    client.stop()
"""

import ctypes
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class InferRequest:
    """Request parameters for a single inference call."""
    token_ids: List[int]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stop_token: int = -1
    vocab_size: int = 32
    kv_size_per_chip: int = 64
    kv_step_size: int = 64


@dataclass
class InferResponse:
    """Response from the inference engine."""
    output_tokens: List[int] = field(default_factory=list)


class TestPathClient:
    """Python wrapper around the TestPath C API (pypto_testpath.so)."""

    # Wire format header: 7 x i32/f32 + 1 x u32 = 32 bytes
    _HEADER_FMT = "<IffiiiiI"
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self._find_lib()
        self._lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self._started = False

    @staticmethod
    def _find_lib() -> str:
        """Search for pypto_testpath.so in common build locations."""
        candidates = [
            Path(__file__).parent.parent.parent / "build" / "pypto_testpath.so",
            Path(__file__).parent.parent.parent / "build" / "libpypto_testpath.so",
            Path(os.getcwd()) / "build" / "pypto_testpath.so",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        raise FileNotFoundError(
            f"Cannot find pypto_testpath.so. Searched: {[str(c) for c in candidates]}")

    def _setup_signatures(self):
        self._lib.testpath_init.restype = ctypes.c_int
        self._lib.testpath_init.argtypes = []

        self._lib.testpath_shutdown.restype = None
        self._lib.testpath_shutdown.argtypes = []

        self._lib.testpath_start.restype = ctypes.c_int
        self._lib.testpath_start.argtypes = [ctypes.c_char_p]

        self._lib.testpath_stop.restype = None
        self._lib.testpath_stop.argtypes = []

        self._lib.testpath_inject_request.restype = ctypes.c_int
        self._lib.testpath_inject_request.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t]

        self._lib.testpath_get_response.restype = ctypes.c_int64
        self._lib.testpath_get_response.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t]

        self._lib.testpath_write_trace.restype = ctypes.c_int
        self._lib.testpath_write_trace.argtypes = []

    def init(self):
        rc = self._lib.testpath_init()
        if rc != 0:
            raise RuntimeError(f"testpath_init failed: {rc}")

    def start(self, trace_path: Optional[str] = None):
        """Initialize and start the serving engine."""
        self.init()
        tp = trace_path.encode() if trace_path else None
        rc = self._lib.testpath_start(tp)
        if rc != 0:
            raise RuntimeError(f"testpath_start failed: {rc}")
        self._started = True

    def stop(self):
        """Stop the serving engine and release resources."""
        if self._started:
            self._lib.testpath_stop()
            self._lib.testpath_shutdown()
            self._started = False

    def inject_request(self, req: InferRequest) -> None:
        """Serialize and inject a request into the engine."""
        header = struct.pack(
            self._HEADER_FMT,
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.stop_token,
            req.vocab_size,
            req.kv_size_per_chip,
            req.kv_step_size,
            len(req.token_ids),
        )
        tokens = struct.pack(f"<{len(req.token_ids)}Q", *req.token_ids)
        buf = header + tokens

        cbuf = ctypes.create_string_buffer(buf)
        rc = self._lib.testpath_inject_request(cbuf, len(buf))
        if rc != 0:
            raise RuntimeError(f"testpath_inject_request failed: {rc}")

    def get_response(self) -> InferResponse:
        """Block until the response is ready and return parsed tokens."""
        resp_buf = ctypes.create_string_buffer(65536)
        nbytes = self._lib.testpath_get_response(resp_buf, 65536)
        if nbytes < 0:
            raise RuntimeError(f"testpath_get_response failed: {nbytes}")

        raw = resp_buf.raw[:nbytes]
        n_tokens = struct.unpack_from("<I", raw, 0)[0]
        tokens = list(struct.unpack_from(f"<{n_tokens}Q", raw, 4))
        return InferResponse(output_tokens=tokens)

    def infer(self, token_ids: List[int], **kwargs) -> InferResponse:
        """Convenience: inject request and get response in one call."""
        req = InferRequest(token_ids=token_ids, **kwargs)
        self.inject_request(req)
        return self.get_response()

    def write_trace(self) -> bool:
        """Write Perfetto trace file. Returns True if written."""
        return self._lib.testpath_write_trace() == 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
