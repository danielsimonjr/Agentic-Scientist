export interface AgentConfig {
  name: string;
  model?: string | null;
  models?: Readonly<Record<string, string | null>>;
  pool?: readonly string[];
}
export interface AgentSelection { agent: string; model: string | null }

export function resolveAgentModel(config: AgentConfig, agent: string): string | null {
  return config.models?.[agent]?.trim() || config.model?.trim() || null;
}

/** Persist each randomized assignment so resuming does not change the harness. */
export function resolveCycleAgent(config: AgentConfig, state: Record<string, unknown>, cycleIndex: number, random: () => number = Math.random): AgentSelection {
  if (!config.pool?.length) return { agent: config.name, model: resolveAgentModel(config, config.name) };
  let raw = state.agent_cycle_assignments;
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    raw = {};
    state.agent_cycle_assignments = raw;
  }
  const assignments = raw as Record<string, unknown>, key = String(cycleIndex);
  let agent = assignments[key];
  if (typeof agent !== "string" || !agent.trim()) {
    const sample = random();
    if (!Number.isFinite(sample) || sample < 0 || sample >= 1) throw new RangeError("Random sample must be in [0,1)");
    agent = config.pool[Math.floor(sample * config.pool.length)]!;
    assignments[key] = agent;
  }
  return { agent: agent as string, model: resolveAgentModel(config, agent as string) };
}

export function splitCycleBudget(totalCycles: number, workers: number): number[] {
  if (!Number.isSafeInteger(totalCycles) || totalCycles < 0 || !Number.isSafeInteger(workers) || workers < 1) throw new RangeError("Expected nonnegative cycles and positive workers");
  const base = Math.floor(totalCycles / workers), remainder = totalCycles % workers;
  return Array.from({ length: workers }, (_, i) => base + (i < remainder ? 1 : 0));
}
