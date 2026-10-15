import type { ComputeBackend, Tensor } from "./types.js";

type Edge = { parent: Variable; backward: (gradient: Tensor) => Tensor };

export class Variable {
  gradient: Tensor | undefined;
  constructor(readonly tape: Tape, readonly value: Tensor, readonly edges: readonly Edge[]) {}
  add(other: Variable): Variable { return this.tape.add(this, other); }
  sub(other: Variable): Variable { return this.tape.sub(this, other); }
  mul(other: Variable): Variable { return this.tape.mul(this, other); }
  matmul(other: Variable): Variable { return this.tape.matmul(this, other); }
  sum(): Variable { return this.tape.sum(this); }
}

/** Reverse-mode autodiff. A tape owns intermediate values and gradients, never
 * external parameter tensors. Dispose each tape after a training step. */
export class Tape {
  private owned = new Set<Tensor>();
  private variables: Variable[] = [];
  private disposed = false;
  private backwardCalled = false;
  constructor(readonly backend: ComputeBackend) {}
  private check(...variables: Variable[]): void {
    if (this.disposed) throw new Error("Tape has been disposed");
    if (this.backwardCalled) throw new Error("A tape supports one backward pass; create a new tape for the next step");
    for (const variable of variables) if (variable.tape !== this) throw new Error("Cannot mix autodiff tapes");
  }
  private own(tensor: Tensor): Tensor { this.owned.add(tensor); return tensor; }
  private node(value: Tensor, edges: Edge[], owned = true): Variable {
    if (owned) this.own(value);
    const variable = new Variable(this, value, edges);
    this.variables.push(variable);
    return variable;
  }
  variable(value: Tensor): Variable {
    this.check();
    if (value.backend !== this.backend) throw new Error("Parameter belongs to another backend");
    return this.node(value, [], false);
  }
  constant(data: ArrayLike<number>, shape: readonly number[]): Variable {
    this.check();
    return this.node(this.backend.tensor(data, shape), []);
  }
  add(a: Variable, b: Variable): Variable {
    this.check(a, b);
    return this.node(this.backend.binary("add", a.value, b.value), [
      { parent: a, backward: g => g }, { parent: b, backward: g => g },
    ]);
  }
  sub(a: Variable, b: Variable): Variable {
    this.check(a, b);
    return this.node(this.backend.binary("sub", a.value, b.value), [
      { parent: a, backward: g => g }, { parent: b, backward: g => this.own(this.backend.unary("neg", g)) },
    ]);
  }
  mul(a: Variable, b: Variable): Variable {
    this.check(a, b);
    return this.node(this.backend.binary("mul", a.value, b.value), [
      { parent: a, backward: g => this.own(this.backend.binary("mul", g, b.value)) },
      { parent: b, backward: g => this.own(this.backend.binary("mul", g, a.value)) },
    ]);
  }
  matmul(a: Variable, b: Variable): Variable {
    this.check(a, b);
    return this.node(this.backend.matmul(a.value, b.value), [
      { parent: a, backward: g => this.own(this.backend.matmul(g, this.own(this.backend.transpose(b.value)))) },
      { parent: b, backward: g => this.own(this.backend.matmul(this.own(this.backend.transpose(a.value)), g)) },
    ]);
  }
  exp(a: Variable): Variable {
    this.check(a);
    const value = this.backend.unary("exp", a.value);
    return this.node(value, [{ parent: a, backward: g => this.own(this.backend.binary("mul", g, value)) }]);
  }
  log(a: Variable): Variable {
    this.check(a);
    return this.node(this.backend.unary("log", a.value), [{ parent: a, backward: g => this.own(this.backend.binary("div", g, a.value)) }]);
  }
  sum(a: Variable): Variable {
    this.check(a);
    return this.node(this.backend.sum(a.value), [{ parent: a, backward: g => {
      // Broadcast on the selected backend without reading the gradient to JS.
      return this.own(this.backend.broadcastScalar(g, a.value.shape));
    } }]);
  }
  backward(loss: Variable): void {
    this.check(loss);
    if (loss.value.size !== 1) throw new RangeError("Backward requires a scalar loss");
    this.backwardCalled = true;
    loss.gradient = this.own(this.backend.tensor([1], loss.value.shape));
    for (let i = this.variables.length - 1; i >= 0; i--) {
      const variable = this.variables[i]!;
      if (!variable.gradient) continue;
      for (const edge of variable.edges) {
        const contribution = edge.backward(variable.gradient);
        edge.parent.gradient = edge.parent.gradient
          ? this.own(this.backend.binary("add", edge.parent.gradient, contribution))
          : contribution;
      }
    }
  }
  dispose(): void {
    for (const tensor of this.owned) tensor.dispose();
    this.owned.clear();
    for (const variable of this.variables) variable.gradient = undefined;
    this.variables.length = 0;
    this.disposed = true;
  }
}

/** Explicit parameter update; caller replaces and disposes the previous tensor. */
export function sgd(parameter: Variable, learningRate: number): Tensor {
  if (!Number.isFinite(learningRate) || learningRate <= 0) throw new RangeError("Learning rate must be positive and finite");
  if (!parameter.gradient) throw new Error("Run backward before updating a parameter");
  const backend = parameter.value.backend;
  const scale = backend.tensor(new Float32Array(parameter.value.size).fill(learningRate), parameter.value.shape);
  let step: Tensor | undefined;
  try {
    step = backend.binary("mul", parameter.gradient, scale);
    return backend.binary("sub", parameter.value, step);
  } finally { scale.dispose(); step?.dispose(); }
}
