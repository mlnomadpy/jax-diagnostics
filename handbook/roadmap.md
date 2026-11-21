# Roadmap

The current release covers the public JAX exception catalog and common backend symptoms, with a Python annotation wrapper, a CLI, and an agent skill. The work below is proposed.

| Priority | Gap | What would count as done |
| --- | --- | --- |
| 1 | Symbolic symptoms can match unrelated messages. The current test corpus is small. | Collect redacted real tracebacks across JAX versions and backend types. Include ambiguous and unrelated cases, and measure correct routes and abstentions. |
| 2 | Chained failures and distributed logs can hide the useful cause. | Add structured cause and exception-group handling, then test nested wrappers, independent branches, and mixed rank logs. Preserve original failure behavior. |
| 3 | Flax, Orbax, and Pallas share broad symptom routes. | Add narrower routes only where examples establish a distinctive trigger and a verified repair. Track the version and backend for each fixture. |
| 4 | Links default to a moving branch. | Offer a release-pinned guide base and verify that every built release resolves all diagnostic anchors. |
| 5 | A worker can exit before Python surfaces an exception. | Evaluate a launcher or log integration that preserves exit status and distinguishes process-level failures from Python exceptions. |

Start with a representative failure and a nearby case that must stay unmatched. Prefer adding evidence to expanding keyword lists.
