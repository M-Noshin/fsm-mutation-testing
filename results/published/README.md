# Published results

Result workbooks backing the figures in the paper. Everything else in
`results/` is gitignored, so put anything worth keeping here.

Each row records the seed, kill condition, depth cap and mutant file it came
from, plus the input sequence found, so any figure can be traced back to a
reproducible run and re-verified:

```bash
python -m fsm_mutation.validate results/published/<file>.xlsx --data-dir data
```
