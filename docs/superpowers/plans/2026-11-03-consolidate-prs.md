# Consolidate historical PRs implementation plan

Goal: submit one new reviewable Draft PR against current main, integrating closed,
unmerged PRs #1–#4. User will review and merge; no merge or PLAXIS calculation here.
Spec: user request on 2026-11-03 plus the compatibility and recovery boundaries of #1–#4.
Architecture: squash all four branch deltas onto main, then test the combined installed
package and remove integration defects. Preserve per-PR provenance and pending live gates.
Tech stack: Python 3.9/3.11, setuptools wheel/sdist, unittest/pytest, GitHub Actions.

## Source heads (verified)
- main: 8a30dfe2604fab16343e2604b9d667ba9b905786
- #1: afb939d282168e7e98d3ab43f1c9820c0622a15d (phase convergence + optional IFC import)
- #2: 37d3734b949afc48597d8635b7c1d3010fdc5c52 (read-only V22 probe)
- #3: 87a1e6a8e002e7bcfcbe5a84e58f344760239e65 (packaging + environment credentials)
- #4: 3d62315431adc89f150b3b90563a472074685ef7 (compensated phase writes + Output reads)

## Tasks
1. [x] Inspect all four PRs and current main; squash deltas on a new integration branch.
2. [x] Run existing no-license tests and installed-package regressions. Add RED/GREEN
   tests for combined optional-client behavior, client error propagation, empty phase
   plans and installed-wheel API imports. Fix only integration defects exposed.
3. [x] Replace obsolete branch-specific workflows with an integrated Python 3.9/3.11
   workflow on compatible Ubuntu image; build wheel and sdist and import public API
   without a vendor client. Run whole repository pytest and report any legacy failures.
4. [x] Independent review; fix substantive findings with regression evidence.
5. [ ] Publish new branch and Draft PR, document source coverage, current test results
   and real Windows V22 checks still NOT RUN. Confirm CI and leave merge to user.

## Review focus
- Optional client dependencies must remain optional at import time; missing clients
  fail clearly only when connecting, and broken installed clients must not be masked.
- Empty or entirely disabled phase plans must not start calculate().
- State validation and prestress compensation must survive integrated imports.
- Tests/CI must exercise installed imports, not only AST extraction.
- Source credentials must use environment; never print or add credential values.
- Distinguish no-license CI from live V22 compatibility and mechanical validation.

## Execution evidence
- Added installed-public-API tests; initial imports failed due to missing plxscripting.
  Lazy loading, public exports and empty-plan preflight now pass RED/GREEN regressions.
- Independent review identified swallowed non-ImportError client initialization failures.
  Public Runner.connect regression failed, then passed after loading outside its catch.
- Final local focused suite: Python 3.9/3.11 each 76 passed (3.11: 35 subtests).
- Both wheel/sdist builds pass; outside-checkout installed-wheel imports pass without client.
- Full legacy suite remains red and is disclosed explicitly, with non-gating CI audit.
- Ruling: preserve all contributed behavior, replace stale workflows with one integrated
  workflow, retain strict OK response and Draft pending V22 evidence. No merge requested.
