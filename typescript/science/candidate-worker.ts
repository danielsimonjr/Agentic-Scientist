import { pathToFileURL } from "node:url";

const path = process.argv[2];
if (!path || !process.send) throw new Error("Candidate worker must be started with IPC and a program path");
try {
  const program: unknown = await import(pathToFileURL(path).href);
  if (!program || typeof program !== "object" || !("solution" in program) || typeof program.solution !== "function") {
    throw new TypeError("Candidate must export a solution() function");
  }
  const result: unknown = await program.solution();
  process.send({ ok: true, result });
} catch (error) {
  process.send({ ok: false, error: error instanceof Error ? error.stack ?? error.message : String(error) });
}
