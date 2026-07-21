# Mutant files

Generated mutant sets are not committed -- a 200 000-mutant file for the
128-state machines is hundreds of megabytes. Regenerate them instead; the
generator is seeded per machine, so a given seed reproduces the same set
exactly.

```bash
python -c "
from fsm_mutation.generate import generate_file
for n in (10000, 30000, 50000, 100000, 200000):
    generate_file('data/fsms/128state_fsms_complete.txt',
                  f'data/mutants/complete_{n}.txt',
                  count=n, fault_type=3, faults_per_mutant=3, seed=42)"
```

Naming must be `<machine-class>_<count>.txt` for the runner to find a file
without an explicit `--mutant-file`.
