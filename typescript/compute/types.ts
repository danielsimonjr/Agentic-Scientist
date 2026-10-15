/** Dense row-major float32 tensors. Backends never silently switch devices. */
export interface Tensor {
  readonly shape: readonly number[];
  readonly size: number;
  readonly backend: ComputeBackend;
  dispose(): void;
}

export type UnaryOperation = "neg" | "exp" | "log" | "sqrt";
export type BinaryOperation = "add" | "sub" | "mul" | "div";

export interface ComputeBackend {
  readonly kind: "cpu" | "webgpu";
  tensor(data: ArrayLike<number>, shape: readonly number[]): Tensor;
  read(tensor: Tensor): Promise<Float32Array>;
  unary(operation: UnaryOperation, tensor: Tensor): Tensor;
  binary(operation: BinaryOperation, left: Tensor, right: Tensor): Tensor;
  matmul(left: Tensor, right: Tensor): Tensor;
  transpose(tensor: Tensor): Tensor;
  softmax(tensor: Tensor): Tensor;
  sum(tensor: Tensor): Tensor;
  broadcastScalar(tensor: Tensor, shape: readonly number[]): Tensor;
  dispose(): void;
}

export function elementCount(shape: readonly number[]): number {
  let size = 1;
  for (const dimension of shape) {
    if (!Number.isSafeInteger(dimension) || dimension < 1) {
      throw new RangeError("Tensor dimensions must be positive safe integers");
    }
    size *= dimension;
    if (!Number.isSafeInteger(size) || size > 0x3fffffff) {
      throw new RangeError("Tensor exceeds supported size");
    }
  }
  return size;
}

export function equalShape(left: Tensor, right: Tensor): void {
  if (left.shape.length !== right.shape.length || left.shape.some((d, i) => d !== right.shape[i])) {
    throw new RangeError("Tensor shapes must match (broadcasting is not implicit)");
  }
}

export function matrixShape(tensor: Tensor): [number, number] {
  if (tensor.shape.length !== 2) throw new RangeError("Expected a rank-two tensor");
  return [tensor.shape[0]!, tensor.shape[1]!];
}
