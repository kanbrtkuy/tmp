# Fable5 Round-2 Addendum — Post-Fix Confirmation at f51fe6a

Scope: scoped confirmation of M-1 closure and committed-report accuracy only. No files modified.

## 1. Executed evidence

| Command | Result |
|---|---|
| `git rev-parse HEAD` + `git status --porcelain` | `f51fe6a527eb49c74053dedb6236ff42703b0f0f`, working tree clean |
| `git log --oneline 6c1171c..f51fe6a` | exactly 1 commit: `f51fe6a Close Fable5 snapshot hardening finding` |
| `git diff --stat 6c1171c..f51fe6a` | 3 files, +103/−0: `FABLE5_FULL_CODE_REVIEW_ROUND2.md` +87, `full_sft_runtime.py` +7, `tests/test_stage2_full_sft_runtime.py` +9 |
| `pytest tests/test_stage2_full_sft_runtime.py -o pythonpath=src -q` | **23 passed** |
| Targeted rerun of the two relevant tests in isolation | **2 passed** |
| Broad suite (`pytest tests/ -o pythonpath=src -q`), two runs | run 1: 2 failed / 325 passed / 13 skipped; run 2: 1 failed / 326 passed / 13 skipped (baseline — only the known pre-existing Stage1 macOS `/tmp`→`/private/tmp` failure). See §4. |

## 2. M-1 is fully closed — all four criteria confirmed in the diff and by executed tests

1. **`.jinja` is loadable and rejected.** `_LOADABLE_MODEL_SUFFIXES` now includes `".jinja"`; the extended test writes `chat_template.jinja` into an otherwise-approved snapshot and asserts `FullSFTRuntimeError` "unapproved top-level loadable". Executed: passes.
2. **Every unapproved top-level directory is rejected.** New `elif path.is_dir(): unexpected_loadable.append(path.name)` branch rejects any non-manifest top-level directory (covers `additional_chat_templates/` and any future directory-borne assets); symlinked directories were already caught by the preceding `is_symlink()` branch, and a directory named like an approved file fails the per-manifest-entry regular-file rehash. Executed: the extended test `mkdir`s `additional_chat_templates` and asserts rejection; passes.
3. **Approved README remains allowed.** The fixture still writes `README.md` (non-loadable, non-directory) and verification succeeds with `runtime_file_count == 2`. Executed: passes.
4. **Manifest semantics unchanged.** The 7-line runtime diff touches only the suffix tuple and the scan loop; `APPROVED_MODEL_RUNTIME_FILES`, the hashing scheme, the manifest JSON, and the pinned `CANONICAL_APPROVED_MODEL_MANIFEST_SHA256` (`2edaed78…08f4e1`) are byte-for-byte untouched.

## 3. Committed Round-2 report is accurate

`review-stage/stage2_stage4_full_sft_2xa100_260714/FABLE5_FULL_CODE_REVIEW_ROUND2.md` (87 lines, added by f51fe6a) was read in full and is a verbatim record of my delivered Round-2 report: same 7-part structure, M-1 and m-1..m-4 as stated, `APPROVE_EMBEDDING_FP32_PAGED_OVERRIDE`, PASS verdicts for resume/storage/provenance and layer-grid/sample-adequacy, the six professor-question statuses, and the final line `APPROVE_TO_RUN`. No omissions or alterations.

## 4. Honest disclosure — one-off flake in a pre-existing test (not a blocker, not related to f51fe6a)

`test_resume_rehydration_replaces_each_large_state_and_fresh_branch_is_noop` failed once in my first broad-suite run, then passed on rerun, in module isolation (23/23), and in targeted isolation. The f51fe6a test delta (+9 lines) is in a different test function, so this is pre-existing. Mechanism (from source, `tests/test_stage2_full_sft_runtime.py:440-445, 471-475`): the test stores raw `id()` values of old fake paged tensors, drops the references, then asserts id-set disjointness/equality against newly allocated objects — CPython can recycle a freed object's address, spuriously intersecting the sets. This is a test-harness determinism nit only; the production rehydration path checks identity on live references. Recommended (non-blocking): hold live references to the old tensors for the duration of the assertions.

## 5. Verdict

The sole major nonblocker M-1 is fully closed with executed evidence; the committed report is accurate; no new findings at f51fe6a warrant escalation.

APPROVE_TO_RUN
