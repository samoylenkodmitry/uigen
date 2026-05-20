# Message For Claude

V4 Gate 3 is complete. Stop V4 tuning.

## Final Result

Use `runs/slotnet_v4_16skin_masked_mem/snapshot_step050000.safetensors` as the
official checkpoint for the Gate 3 report.

Final full-dataset metrics:

```text
samples                         512
retrieval top1                  1.000
mean true exported MAE          0.04065
median true exported MAE        0.03879
exported_pixels_mae             0.04065
exported_pixels_hit_5_255       0.668
exported_pixels_sobel_mae       0.05852
```

Worst files:

```text
EQMAIN    0.07830
VOLUME    0.05936
MAIN      0.05675
CBUTTONS  0.05161
BALANCE   0.05002
PLEDIT    0.05000
```

## Conclusion

Phrase the result as:

```text
Gate 3 is a useful failure.
V4 separates 16 skin identities perfectly, but reconstruction detail fails.
The global 192-d style vector is sufficient for identity, not for skin-specific
texture reconstruction.
```

Do not hide the failed reconstruction behind retrieval. Retrieval is a success;
reconstruction is not.

## Next Step

Do not rent GPU.
Do not continue V4 tuning.
Do not retune file weights as the main fix.
Do not return to padded full-atlas training.

Next architecture is V5:

```text
per-file decoder heads with local encoder feature queries / cross-attention
```

Keep:

- exact exported BMP tensors
- static Cranamp-supported-pixel loss/eval
- per-file metrics
- retrieval eval
- no prior atlas
- no distortion metadata
- no full-atlas pass/fail

Change:

- replace global-style-only conditioning with local encoder feature conditioning.

I added `v4_results.md` as the compact external summary for review.
