You are working inside the repository `musclemap-bench`.

Goal:
Build the benchmarking part of a 3-repo thesis system:
- `musclemap-data`
- `musclemap-model`
- `musclemap-bench`

Rules:
1. Never silently invent Kinesis or MotionGPT APIs.
2. If integration details depend on local code, inspect the repository first, then implement.
3. Keep all code deterministic, typed, path-safe, and restart-safe.
4. Preserve the benchmark protocol:
   - Layer 1: MuscleMAP vs Kinesis vs OpenSim
   - Layer 2: MuscleMAP vs MotionGPT
   - Resource comparison
5. Prefer small, verifiable edits.
6. After each implementation task, run tests.

Your first job before writing code:
- inspect the local repositories configured in `config.yaml`
- confirm actual APIs and file structure
- then replace `NotImplementedError` blocks precisely
