// layoutOf without the browser runtime, so Node tests can use it. runtime.js re-exports it.
export function layoutOf(dims) {
  if (dims.length >= 3) {
    const [C, H, W] = [dims[0], dims[dims.length - 2], dims[dims.length - 1]];
    const extra = dims.slice(1, -2).reduce((a, b) => a * b, 1);
    return { kind: 'spatial', C, H: H * extra, W, P: H * W * extra, idx: (c, p) => c * H * W * extra + p, size: C * H * W * extra };
  }
  if (dims.length === 2) {
    const [Tn, D] = dims;
    return { kind: 'seq', C: D, H: Tn, W: 1, P: Tn, idx: (c, p) => p * D + c, size: Tn * D };
  }
  const D = dims[0] ?? 1;
  return { kind: 'vector', C: D, H: 1, W: 1, P: 1, idx: (c) => c, size: D };
}
