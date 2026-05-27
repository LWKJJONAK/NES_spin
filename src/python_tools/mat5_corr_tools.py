#!/usr/bin/env python3
"""Small MAT v5 helpers for correlation files written by rbm.cc/matio."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np


MI_INT8 = 1
MI_UINT8 = 2
MI_INT32 = 5
MI_UINT32 = 6
MI_DOUBLE = 9
MI_MATRIX = 14
MI_COMPRESSED = 15


def _align8(n: int) -> int:
    return (n + 7) & ~7


def _read_tag(buf: bytes, pos: int) -> tuple[int, int, int]:
    word0, word1 = struct.unpack_from("<II", buf, pos)
    small_type = word0 & 0xFFFF
    small_size = word0 >> 16
    if small_size:
        return small_type, small_size, pos + 4
    return word0, word1, pos + 8


def _read_element(buf: bytes, pos: int) -> tuple[int, bytes, int]:
    dtype, nbytes, data_pos = _read_tag(buf, pos)
    data = buf[data_pos : data_pos + nbytes]
    if dtype == MI_COMPRESSED:
        next_pos = data_pos + nbytes
    elif pos + 4 == data_pos:
        next_pos = pos + 8
    else:
        next_pos = data_pos + _align8(nbytes)
    return dtype, data, next_pos


def _decode_ints(dtype: int, data: bytes) -> list[int]:
    if dtype == MI_INT32:
        fmt = "<" + "i" * (len(data) // 4)
    elif dtype == MI_UINT32:
        fmt = "<" + "I" * (len(data) // 4)
    else:
        raise ValueError(f"unsupported integer dtype {dtype}")
    return list(struct.unpack(fmt, data))


def _decode_name(dtype: int, data: bytes) -> str:
    if dtype not in (MI_INT8, MI_UINT8):
        raise ValueError(f"unsupported name dtype {dtype}")
    return data.decode("latin1")


def _read_mat5_stream(buf: bytes, wanted: set[str] | None = None, pos: int = 0) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    while pos < len(buf):
        dtype, matrix_data, pos = _read_element(buf, pos)
        if dtype == MI_COMPRESSED:
            out.update(_read_mat5_stream(zlib.decompress(matrix_data), wanted, 0))
            continue
        if dtype != MI_MATRIX:
            continue

        mpos = 0
        _, _, mpos = _read_element(matrix_data, mpos)  # array flags
        dim_type, dim_data, mpos = _read_element(matrix_data, mpos)
        dims = _decode_ints(dim_type, dim_data)
        name_type, name_data, mpos = _read_element(matrix_data, mpos)
        name = _decode_name(name_type, name_data)

        real_type, real_data, mpos = _read_element(matrix_data, mpos)
        if wanted is not None and name not in wanted:
            continue
        if real_type != MI_DOUBLE:
            raise ValueError(f"variable {name} has unsupported dtype {real_type}")

        arr = np.frombuffer(real_data, dtype="<f8").copy()
        out[name] = arr.reshape(tuple(dims), order="F")

    return out


def read_mat5_arrays(path: Path, wanted: set[str] | None = None) -> dict[str, np.ndarray]:
    buf = path.read_bytes()
    return _read_mat5_stream(buf, wanted, 128)


def summarize(path: Path) -> str:
    data = read_mat5_arrays(path, {"z", "zz"})
    if "zz" not in data:
        return f"{path}: no zz"
    zz = data["zz"]
    if zz.ndim == 2:
        zz = zz[:, :, None]
    z = data.get("z")
    if z is None:
        z_abs = np.nan
        conn_std = np.nan
    else:
        if z.ndim == 1:
            z = z[:, None]
        connected = np.empty_like(zz)
        for k in range(zz.shape[2]):
            connected[:, :, k] = zz[:, :, k] - np.outer(z[:, k], z[:, k])
        mask = ~np.eye(zz.shape[0], dtype=bool)
        conn_std = np.nanstd(connected[mask, :])
        z_abs = np.nanmean(np.abs(z))

    mask = ~np.eye(zz.shape[0], dtype=bool)
    off = zz[mask, :]
    diag_mean = [float(np.nanmean(np.diag(zz[:, :, k]))) for k in range(zz.shape[2])]
    return (
        f"{path}: zz={zz.shape}, diag_mean={np.round(diag_mean, 4).tolist()}, "
        f"off_min={np.nanmin(off):.6g}, off_max={np.nanmax(off):.6g}, "
        f"off_mean={np.nanmean(off):.6g}, off_std={np.nanstd(off):.6g}, "
        f"mean_abs_z={z_abs:.6g}, connected_off_std={conn_std:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.paths:
        print(summarize(path))


if __name__ == "__main__":
    main()
