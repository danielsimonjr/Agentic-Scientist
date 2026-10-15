import { elementCount, equalShape, matrixShape } from "./types.js";
import type { BinaryOperation, ComputeBackend, Tensor, UnaryOperation } from "./types.js";

class CpuTensor implements Tensor {
  readonly size: number;
  readonly shape: readonly number[];
  data: Float32Array | undefined;
  constructor(readonly backend: CpuBackend, data: Float32Array, shape: readonly number[]) {
    this.shape = Object.freeze([...shape]);
    this.size = data.length;
    this.data = data;
  }
  dispose(): void { this.data = undefined; this.backend.release(this); }
}

/** Deterministic float32 reference backend, explicitly selected by the caller. */
export class CpuBackend implements ComputeBackend {
  readonly kind = "cpu";
  private disposed = false;
  private tensors = new Set<CpuTensor>();

  tensor(data: ArrayLike<number>, shape: readonly number[]): Tensor {
    if (this.disposed) throw new Error("Backend has been disposed");
    if (data.length !== elementCount(shape)) throw new RangeError("Data length does not match shape");
    const result = new CpuTensor(this, Float32Array.from(data), shape);
    this.tensors.add(result);
    return result;
  }
  release(tensor: CpuTensor): void { this.tensors.delete(tensor); }
  private data(tensor: Tensor): Float32Array {
    if (this.disposed) throw new Error("Backend has been disposed");
    if (!(tensor instanceof CpuTensor) || tensor.backend !== this) throw new TypeError("Tensor belongs to another backend");
    if (!tensor.data) throw new Error("Tensor has been disposed");
    return tensor.data;
  }
  async read(tensor: Tensor): Promise<Float32Array> { return this.data(tensor).slice(); }
  unary(operation: UnaryOperation, tensor: Tensor): Tensor {
    const functions = { neg: (x: number) => -x, exp: Math.exp, log: Math.log, sqrt: Math.sqrt };
    return this.tensor(this.data(tensor).map(functions[operation]), tensor.shape);
  }
  binary(operation: BinaryOperation, left: Tensor, right: Tensor): Tensor {
    equalShape(left, right);
    const a = this.data(left), b = this.data(right);
    const functions = {
      add: (x: number, y: number) => x + y, sub: (x: number, y: number) => x - y,
      mul: (x: number, y: number) => x * y, div: (x: number, y: number) => x / y,
    };
    return this.tensor(a.map((x, i) => functions[operation](x, b[i]!)), left.shape);
  }
  matmul(left: Tensor, right: Tensor): Tensor {
    const [m, k] = matrixShape(left), [rk, n] = matrixShape(right);
    if (k !== rk) throw new RangeError("Matrix inner dimensions do not match");
    const a = this.data(left), b = this.data(right), result = new Float32Array(m * n);
    for (let row = 0; row < m; row++) for (let col = 0; col < n; col++) {
      let value = 0;
      for (let j = 0; j < k; j++) value = Math.fround(value + Math.fround(a[row * k + j]! * b[j * n + col]!));
      result[row * n + col] = value;
    }
    return this.tensor(result, [m, n]);
  }
  transpose(tensor: Tensor): Tensor {
    const [m, n] = matrixShape(tensor), input = this.data(tensor), output = new Float32Array(tensor.size);
    for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) output[c * m + r] = input[r * n + c]!;
    return this.tensor(output, [n, m]);
  }
  softmax(tensor: Tensor): Tensor {
    const [rows, cols] = matrixShape(tensor), input = this.data(tensor), output = new Float32Array(tensor.size);
    for (let row = 0; row < rows; row++) {
      const offset = row * cols;
      let max = -Infinity, sum = 0;
      for (let c = 0; c < cols; c++) max = Math.max(max, input[offset + c]!);
      for (let c = 0; c < cols; c++) {
        output[offset + c] = Math.exp(input[offset + c]! - max);
        sum += output[offset + c]!;
      }
      for (let c = 0; c < cols; c++) output[offset + c] = output[offset + c]! / sum;
    }
    return this.tensor(output, tensor.shape);
  }
  sum(tensor: Tensor): Tensor {
    let result = 0;
    for (const value of this.data(tensor)) result = Math.fround(result + value);
    return this.tensor([result], []);
  }
  broadcastScalar(tensor: Tensor, shape: readonly number[]): Tensor {
    if (tensor.size !== 1) throw new RangeError("Expected a scalar");
    return this.tensor(new Float32Array(elementCount(shape)).fill(this.data(tensor)[0]!), shape);
  }
  dispose(): void {
    for (const tensor of this.tensors) tensor.dispose();
    this.disposed = true;
  }
}
