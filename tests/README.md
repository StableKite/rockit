# Candidate regression suite (I69)

Run from a clean checkout:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure --no-tests=error
```

The six default Linux/Git-checkout tests cover valid public MPI shared/static
calls, private reconstructed pool/executor/fault state, checked BMP header I/O,
exact core export **names**, and tracked-tip software-binary exclusion.
Python checks are registered only with Python; the Git check requires a Git
checkout. BMP I/O needs MPI examples enabled. C tests use the existing example
warning policy because unmodified vendor headers rely on GNU enum extensions.
No production warning is suppressed. Assertions remain active in Release
because test checks are explicit branches, not `assert`.

They do **not** prove vendor-public C++ ABI, RTMessage implementation, static
archive equivalence, feature behavior, video pipelines, all MPI functions,
32-bit/RK3308, or final hardware/resource qualification. Never turn six green
smoke tests into PROJECT_COMPLETE. See the canonical I69 Drive requirements
ledger and recovery handoff for every remaining gate.

I69 fixed checked reads in three example sources after an actual GCC13
AArch64 `-Werror=unused-result` failure (baseline retained in evidence).
The library core, original public headers and export maps are unchanged.
Fortified GCC/Clang and Clang ASan+UBSan builds exercise the suite in CI.
Local/board results and hosted CI results are separate evidence scopes.
