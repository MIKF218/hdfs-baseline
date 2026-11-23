# Group B, first run (22 Nov 2026): not valid, kept for the record

These results failed the no-drift go/no-go check (F1 0.098 on the test set) and must not be used.
Three problems were found, all fixed in the current `run_drift_b.py`:

1. Numerical: the clean counts were stored as 32-bit floats, which corrupted the covariance used by PCA.
2. File paths were not masked, so one allocateBlock event appeared as 63 different "messages"
   (Drain3 treats paths as wildcards in group A).
3. No TF-IDF weighting, unlike group A.

The fixes were chosen from the no-drift validation split and from parity with group A, not from these
drift results.
