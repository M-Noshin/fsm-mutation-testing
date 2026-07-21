---
name: Bug report
about: Something produced a wrong or unexpected result
labels: bug
---

**What happened**

**Command you ran**

```
python -m fsm_mutation.run ...
```

**Machine family and mutant set**
Which `--machines`, how many mutants, which fault type, which seed.

**Did the validator agree?**

```
python -m fsm_mutation.validate results/<file>.xlsx --data-dir data
```

Paste the output. If a row reports `MISMATCH`, include the `Best Sequence` and
`Final Mutation Score (%)` from that row — that combination is enough to
reproduce a scoring bug exactly.

**Environment**
Python version, OS, numpy version.
