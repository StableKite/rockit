# Source-only bounded retain-path and comparator evidence

These sources are TEST ONLY. `retain_paths.c` tests candidate ownership/refcount
under `ROCKIT_MPP_PROVIDER=disabled` and `ROCKIT_MMZ_PROVIDER=host`, not
actual hardware, codec, or vendor equivalence. All four `ai/vpss/avs/vdec`
cases must run in isolated processes with finite timeouts and exact source,
release, and runtime hashes. `negative_status.c` is an LD_PRELOAD deliberately
incorrect status shim; pass means that the positive test detects the error.
Do not preload it into real applications or install vendor libraries systemwide.

`../compare_mpi_records.py` compares only the original-vendor-first
1920x1080 YUV420 calibration + 4096-byte MB valid-input five record set,
including return-status and reference-count fields; it also checks process
exit values, duplicates, missing records, and known expected constants.
Full vendor stdout/stderr and input hashes must be retained. No C++ TGI ABI,
invalid input, hardware DMA, or broad MPI equivalence is claimed.

Source provenance: I69/I70 verified evidence in canonical Google Drive
20_EVIDENCE, with source hashes and native/QEMU replay receipts.