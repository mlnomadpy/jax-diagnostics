# Python API

## `annotate_errors(*, guide_base=...)`

The context manager handles exceptions raised out of its block. When a route matches, it adds a note to the existing exception and re-raises it. Exception type, cause, and existing notes remain available. Nested wrappers with the same guide base add the same note once. Interrupts and exits propagate through Python's usual behavior.

```python
with annotate_errors():
    main()
```

Guide URLs default to the repository's `main` branch. Supply a pinned GitHub tree or a mirror whose root contains `references/` to use a stable copy.

The wrapper belongs at a host-side entry point outside JAX transforms. Device errors can surface when asynchronous work synchronizes. Use the original backend and existing synchronization boundary when reproducing the failure. The wrapper leaves scheduling and checkpoints to the application.

## `diagnose(source)`

Pass an exception name, traceback, or symptom string. The function returns a `Diagnosis` object on a match and `None` when no route applies.

```python
from jax_diagnostics import diagnose

result = diagnose("TracerBoolConversionError")
assert result is not None
print(result.id, result.link)
```

A diagnosis exposes `id`, `title`, `guide`, `link`, `evidence`, and `match`. The link is relative to the guide root. A match is either an exact exception route or a symptom route. Symptom evidence is a phrase from the input. Each route is a starting point for diagnosis.
