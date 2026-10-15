# TypeScript rewrite status

**In progress; not a replacement release.** The approved target is a complete
TypeScript rewrite: Bun for development, compiled ESM for Node.js distribution,
and WebGPU/WGSL for GPU workloads. There must be no Python runtime, Python
subprocess bridge, PyTorch, vLLM, or project-specific CUDA implementation in the
finished product.

The original Python application remains in this branch as the behavior reference
until the replacement is complete. The existing `agsci` CLI still runs that
application. Adding the TypeScript package does not migrate that CLI. The npm
package is private while the rewrite is incomplete.

Baseline: commit `2b8aa39285c35aaca3b0e06dd121061ef8a2d854`, 400 Python files and
192,970 lines including tests, examples, and supporting scripts.

## Implemented in this branch

| Component | Implemented behavior | Remaining limitations |
| --- | --- | --- |
| Toolchain | Strict TypeScript, Bun lockfile/tests, emitted Node ESM and declarations, separate compiled test output | Full application packaging and CLI are pending |
| Compute | Shared WebGPU device API; WGSL matrix multiplication, transpose, elementwise operations, softmax, sum, scalar broadcast; explicit CPU reference backend | Initial correctness kernels, not optimized large-model kernels; no implicit broadcasting, tiling or sharding |
| Autodiff | Reverse mode for add/subtract/multiply/matmul/exp/log/sum; shared graph accumulation; SGD | Transformer operators, optimizer states, gradient checkpointing and full GRPO training are pending |
| Circle packing | Original 26-circle constructor, constraints, metrics, killable candidate execution, logs, `correct.json`, `metrics.json`, interoperable `extra.npz` | Not yet connected to task drafting or investigation |
| GRPO arithmetic | Sample-standard-deviation group advantages, clipped policy loss and log-probability derivative | These are reference numerical helpers, not a port of Qwen training or vLLM sampling |
| Pipeline utilities | Run paths and UTC run IDs, persisted cycle-agent selection, worker cycle budgets, bounded billing provenance | These are library ports; the pipeline still uses its original implementation |
| Validation | Golden circle-packing arrays/score, finite-difference gradient checks, analytical matrix gradients, small model training, evaluator failures and timeout tests | Hardware GPU and full end-to-end experiments are not validated |

## Develop and validate

```sh
bun install --frozen-lockfile
bun run check
```

`bun run build` emits ordinary JavaScript and declarations to `dist/`. No Bun
runtime API is imported by the distribution. `bun run test:node` compiles tests
separately and executes them with Node. The CPU tests run without a GPU. The GPU
test is deliberately opt-in and fails, rather than switching to CPU, if no
adapter exists:

```sh
bun run test:webgpu
```

For a headless Linux machine, a Vulkan implementation and working ICD are
required by the Node WebGPU provider. The CI workflow extracts SwiftShader from
`@sparticuz/chromium@153.0.0` into temporary storage and sets `VK_ICD_FILENAMES`.
That runs the actual WGSL shaders through software Vulkan. It establishes shader
correctness, not hardware GPU performance or large-model feasibility.

Browser consumers import `agentic-scientist/compute` and supply a `GPUDevice` or
call `createBrowserBackend()`. Node consumers import
`agentic-scientist/compute/node` and call `createNodeBackend()`. The latter uses
the optional Dawn `webgpu` package, which is a native implementation of the web
standard, not a Python or CUDA application backend. Node backends must be closed
with `close()`; browser-created backends destroy their device on `dispose()`.
Backends constructed with a supplied device leave device ownership with the
caller. Tensors and autodiff tapes have explicit disposal.

Example using the built distribution:

```ts
import { createNodeBackend } from 'agentic-scientist/compute/node';

const backend = await createNodeBackend();
try {
  const a = backend.tensor([1, 2, 3, 4], [2, 2]);
  const b = backend.tensor([1, 0, 0, 1], [2, 2]);
  console.log(await backend.read(backend.matmul(a, b)));
} finally {
  backend.close();
}
```

The standalone circle evaluator is exported as `agentic-scientist/evaluate`.
Candidates export `solution()` from an ES module and return
`[centers, radii, reportedSum]`. Distributed candidates must be JavaScript
compiled from their TypeScript source; Bun can load TypeScript during
development. Subprocess execution is not a security sandbox. This implementation
must be integrated into the container and sealed-evaluation boundary before
untrusted generated programs are accepted by the application.

## Intentional changes from the original evaluator

- Non-finite coordinates/radii are rejected explicitly.
- Invalid candidates receive a zero combined score rather than their reported
  sum. Failed evaluations remove an earlier `extra.npz` in the same output
  directory, so stale successful data cannot appear to belong to the failure.
- Logs are bounded and child execution is killed after a result, timeout or
  log overflow. POSIX process groups are terminated; Windows currently only
  terminates the immediate child and is not yet an approved execution target.
- Invalid tensor shapes, malformed budgets, and one-element sample-standard-
  deviation groups fail explicitly rather than returning undefined/NaN values.
- Named-user home paths such as `~otheruser/path` require an explicit path.

The circle fixture was generated once from the baseline implementation. The new
tests read JSON and do not execute Python. The TypeScript NPZ writer preserves
float64 values and the original `centers`, `radii`, and `reported_sum` names.

## Work required before replacing the application

1. Port configuration composition (Hydra/OmegaConf semantics), presets and the
   `agsci` CLI; preserve override precedence, errors, dry-run and resume behavior.
2. Port process supervision, agent/container launch, authentication forwarding,
   resource ownership, runtime caching, artifact publication and recovery.
   Preserve the existing descriptor-safe filesystem guarantees; a simple
   path-validation/rename replacement is not equivalent.
3. Port task drafting/qualification, investigation, ablations, sealed final
   evaluation, post-hoc analysis, paper/review/revision stages and their tests.
4. Replace the inspector backend and its launch/build path; migrate remaining
   frontend JavaScript helpers and tests into TypeScript/Bun.
5. Port reference retrieval, evidence checks, plotting, PDF handling, scientific
   utilities, shell entry points that call Python, and every example runtime.
6. Implement tokenizer/model-weight loading, transformer forward and backward
   operators, sampling, reward grading, optimizer/checkpoint handling and full
   GRPO training on WebGPU. Validate against small fixed reference fixtures
   before testing the actual Qwen task. Account explicitly for device storage
   limits and memory, precision, and multi-device differences.
7. Validate real hardware GPU execution and full scientific workflows, package
   and install the complete CLI under Node without Bun or Python, then remove
   the original Python files, manifests, lockfiles and container dependencies.

Passing the current tests is **not** the completion criterion for the rewrite.
Do not remove the original application or mark this migration complete until the
remaining behavior has been implemented and validated.

## Implementation references

- [Dawn Node WebGPU provider](https://github.com/dawn-gpu/node-webgpu)
- [Bun TypeScript support](https://bun.sh/docs/typescript)
- [NumPy NPY/NPZ format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
- [fflate ZIP implementation](https://github.com/101arrowz/fflate)
