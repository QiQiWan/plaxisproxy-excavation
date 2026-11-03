# Phase property and prestress recovery implementation plan

Goal: close PR #4 partial-write risks, preserving Draft and never starting calculations.
Architecture: process-local serialization, preflight scalar snapshots, phase-only writes,
combined readback, reverse-order compensation and verified recovery. This is best-effort
compensation, not a native PLAXIS transaction. Unknown recovery must stop caller work.

Constraints: Python 3.9/3.11; no new runtime dependencies; no global setattr fallback;
no calculation/save/open commands in verification script; actual Output force is separate.
The user's detailed Action is the implementation specification and authorizes execution.
Work in this fresh clone on the existing PR branch; no changes to main or other PRs.

1. Add failure-injecting fake-client tests, reproduce second-write partial state.
2. Implement transactional single-property and prestress writes. Snapshot all properties
   before any mutation; catch proxy lookup/index/value failures; strict booleans; no-op
   repeated writes; verify both properties together. Restore even a setter that raises
   after changing state. Continue recovery after individual restore errors and report
   final readback certainty. Optional guard_phases detect contamination without editing
   other stages. Local lock cannot coordinate GUI/other processes; require exclusive access.
3. Add explicit-phase Output reader with explicit result-type handle and confirmed kN unit.
   Reject empty, textual, nonfinite results; preserve individual values without summing.
4. Provide Windows V22 scratch-model probe (manual sign verification, explicit mutation
   opt-in, guaranteed recovery attempt, JSON evidence; Output read is separate/read-only).
5. Run focused tests on 3.9/3.11, full repository pytest, compile probe, inspect diff,
   push only PR branch, confirm new CI and Draft state. Record real V22 as NOT RUN.

Review focus: apply-then-raise; rollback failure; unreadable snapshots; guard contamination;
proxy truthiness/equality; no-op updates; final pair drift; Output phase and force-unit ambiguity.

## Execution record
- Contract tests reproduced original partial-write failures before implementation.
- Independent review reproduced coupled setter side effects, cancellation bypass and
  a false-positive fault probe. Added failing tests and fixed all three; mapping Output
  payloads are now rejected as invalid force data.
- Final local contracts: Python 3.9.25 and 3.11.17 each PASS 41 tests; focused pytest
  PASS 41 tests / 33 subtests. Probe compiles in both Python versions.
- Full pytest BLOCKED: same 11 collection errors on original HEAD and patched tree.
- Ruling: restore every target snapshot property after any attempted write because
  PLAXIS setters may have coupled effects; never write guard phases. Native inheritance
  restoration and external concurrent writers remain outside this scalar compensation.
- Windows licensed V22 and real solver verification NOT RUN. Draft remains required.
