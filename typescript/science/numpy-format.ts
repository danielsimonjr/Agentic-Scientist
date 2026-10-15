import { zipSync } from "fflate";

/** NumPy format v1.0, little-endian float64, C ordering. NPY headers use
 * a Python-literal dictionary but producing them requires no Python runtime. */
export function encodeNpy(data: readonly number[], shape: readonly number[]): Uint8Array {
  let count = 1;
  for (const dimension of shape) {
    if (!Number.isSafeInteger(dimension) || dimension < 0) throw new RangeError("Invalid NPY shape");
    count *= dimension;
    if (!Number.isSafeInteger(count)) throw new RangeError("NPY shape too large");
  }
  if (data.length !== count) throw new RangeError("NPY data does not match shape");
  const tuple = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  const dictionary = `{'descr': '<f8', 'fortran_order': False, 'shape': (${tuple}), }`;
  const padding = (64 - ((10 + dictionary.length + 1) % 64)) % 64;
  const header = new TextEncoder().encode(dictionary + " ".repeat(padding) + "\n");
  if (header.length > 65535) throw new RangeError("NPY v1 header too large");
  const bytes = new Uint8Array(10 + header.length + data.length * 8);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  const view = new DataView(bytes.buffer);
  view.setUint16(8, header.length, true);
  bytes.set(header, 10);
  data.forEach((value, i) => view.setFloat64(10 + header.length + i * 8, value, true));
  return bytes;
}

export function encodeNpz(arrays: Readonly<Record<string, { data: readonly number[]; shape: readonly number[] }>>): Uint8Array {
  const files: Record<string, Uint8Array> = Object.create(null) as Record<string, Uint8Array>;
  for (const [name, array] of Object.entries(arrays)) {
    if (!/^[A-Za-z0-9_-]+$/.test(name)) throw new Error("NPZ array names must be simple identifiers");
    files[`${name}.npy`] = encodeNpy(array.data, array.shape);
  }
  return zipSync(files, { level: 0 });
}
