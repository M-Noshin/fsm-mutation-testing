# Specification files

Drop the FSM specification files here. The runner looks for these names:

| `--machines` | expected file |
|---|---|
| `complete` | `128state_fsms_complete.txt` |
| `partial`  | `128state_fsms_partial.txt` |
| `cerny`    | `cerny_machines.txt` |

Any other file works via `--machines custom --spec-file <path>`.

## Format

A file holds one or more FSMs. Each starts with a header line of six integers:

```
<fsm_id> <num_states> <num_transitions> <num_inputs> <num_outputs> <void>
```

followed by transition lines of four tokens:

```
<source> <target> <input> <output>
```

Blocks are separated by blank lines. Partial machines simply omit the
transitions that are undefined. Input labels may be letters or digits; state
labels must be integers. The initial state is the numerically smallest state
label.

Example:

```
0 2 4 2 2 0
1 2 0 0
1 1 1 1
2 1 0 1
2 2 1 0
```

If a header disagrees with the transitions that follow, the parser warns rather
than trusting the header.
