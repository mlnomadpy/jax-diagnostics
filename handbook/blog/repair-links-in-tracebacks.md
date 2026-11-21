# Why put repair links in tracebacks?

Imagine asking a coding agent to fix a training job that failed on a TPU overnight. It has a traceback and a repository full of Python. It can write a plausible patch in seconds. The harder questions come first. Did the error originate while JAX was tracing Python, compiling a program, or running it on the device? Does the failure reproduce with the same inputs and sharding? What would count as a fix?

An experienced developer starts by answering those questions. We should make that same path easy for an agent to follow, including when the framework or a new API was absent from the model's training data.

## Agent experience is part of developer experience

A library already has interfaces for people who debug it. There are error messages, documentation, examples, and tests. A coding agent uses those surfaces too, often through a terminal and a handful of file reads. If the useful information is split across a long stack trace, a release note, and a forum thread about an older version, the agent has to reconstruct the connection. A fluent explanation can hide the fact that it picked the wrong version or made the failure disappear by changing the execution path.

The interface can change the outcome even when the underlying model stays fixed. **SWE-agent** designed an agent-facing interface for navigating source, editing files, and running commands, then evaluated it on repository-level software repair. Framework maintainers have a similar opportunity at a smaller scale. An error can point directly to the evidence an agent should inspect and the smallest check it should run. That saves an exploratory detour on cases where the framework already knows enough to give good directions. [Yang et al., *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*](https://papers.nips.cc/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html)

## What debugging actually looks like

A useful debugging loop has an observable failure at one end and a verification step at the other.

1. **Capture the failure.** Save the command, original exception chain, versions, backend, and the smallest relevant input. For an accelerator job, also record shape, dtype, and sharding when they affect reproduction. A later synchronization point may be where asynchronous device execution surfaces the error in Python. [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
2. **Pick a hypothesis and say what would disprove it.** If a Python branch is using a traced predicate, a tiny example under the same transform should fail before device computation matters. If output only becomes non-finite under compiled execution on one backend, an eager CPU test has a different verification scope.
3. **Run the cheapest check that distinguishes the possibilities.** Reduce the failing expression while preserving the trigger. Inspect an intermediate value, run a focused test, or compare the same operation under two deliberately chosen configurations. Change one relevant variable at a time.
4. **Patch the cause and rerun the original failure.** Test the nearby case that a tempting workaround would break. For a boolean branch, exercise both outcomes. For a sharding fix, compare outputs as well as placement or memory. Keep the relevant backend and precision when checking a backend-specific result.

Execution gives the model feedback that a plausible explanation alone cannot provide. In *Teaching Large Language Models to Self-Debug*, Chen and colleagues studied an iterative loop that used execution results and explanations to improve generated code in their benchmark settings. Their work motivates keeping the failing result and verification command visible in the agent's context. For training bugs, independently specified checks give that feedback a stronger basis than tests written by the same agent proposing the fix. [Chen et al., *Teaching Large Language Models to Self-Debug*](https://arxiv.org/abs/2304.05128)

## How to help an agent debug a framework it has never seen

Start with the installed version and the failing operation. Give the agent the relevant contract from that version's docs or source, a small example, and a way to check the proposed behavior. A model can reason about an unfamiliar function if it can inspect what arguments the function accepts, what it promises to return, and how it fails. It can then use the runtime to settle questions where documentation and observed behavior disagree.

**DocPrompting** studied this idea for code generation. It retrieved relevant documentation at inference time and used it to guide code for the requested task. The experiments measured code generation. We borrow the retrieval design for a different task, debugging. If an exception class is already known, we can use it as a direct lookup key and save broad document retrieval for cases with weak or ambiguous signals. [Zhou et al., *DocPrompting: Generating Code by Retrieving the Docs*](https://arxiv.org/abs/2207.05987)

The amount of context matters. Start with the exception and the frame that failed, then load a short guide. Open the full compiler docs only if the next hypothesis needs them. Keep one copy of the repro command and record what each experiment ruled out. That gives the agent room to reason about the actual code. It also gives maintainers something to optimize beyond model choice. In a release benchmark, we would measure time to a verified fix, tokens consumed, test runs, and accelerator minutes alongside correctness. [Anthropic, *Effective context engineering for AI agents*](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents)

## Update the guidance without retraining the model

Model weights and framework guidance have different release cycles. A framework maintainer can update a skill when a release changes a contract, when a misleading error becomes common, or when a better reproduction is found. An agent running the same model can load the new guidance on its next task. The model's parameters stay fixed. The information and tools available to it at inference time change.

Agent Skills make this practical through progressive disclosure. A short name and description help the agent decide when a skill applies. The skill's `SKILL.md` explains how to work through the problem. Supporting files carry specific reference material or runnable helpers. Put durable facts and examples in the docs, and use the skill to answer a procedural question such as which evidence to collect first and what would verify the fix. The update loop belongs in normal engineering review. Pin a framework version in the reproduction, review the guide change as code, and keep negative fixtures for symptoms that should stay unmatched. [Anthropic, *Equipping agents for the real world with Agent Skills*](https://www.anthropic.com/research/equipping-agents-for-the-real-world-with-agent-skills)

Skills can also send an agent down the wrong path. SkillsBench reported positive average effects for curated skills with considerable variation, including tasks that performed worse. SWE-Skills-Bench uses paired runs on pinned software repositories with explicit acceptance tests to examine the contribution of a skill. For a diagnostic guide, that suggests testing whether it changes decisions on real failures and recording whether it induces a confident wrong fix. A guide's length or number of install commands tells us little about that. [*SkillsBench*](https://arxiv.org/abs/2602.12670) · [*SWE-Skills-Bench*](https://arxiv.org/abs/2603.15401)

## How JAX Diagnostics applies this

Solid 2.0 provided a concrete example. Its reactivity diagnostics can report a stable code and a link into a repair skill for that code, next to the failing location. The agent can jump directly from the observation to the applicable procedure. We wanted the same affordance for JAX programs through a standalone, opt-in package. [Solid 2.0 announcement](https://www.solidjs.com/blog/solid-2-0-the-long-hold/) · [Solid diagnostics reference](https://docs.solidjs.com/reference/solid-js/diagnostics/)

JAX Diagnostics uses the original exception as evidence and adds a stable ID with a repair link. The Python wrapper attaches a note and re-raises the exception. The CLI handles saved text, and the agent skill points into the same guides. Exact exception classes get specific routes. A generic status such as `RESOURCE_EXHAUSTED` can suggest the memory guide, which then asks whether the failure involved per-device allocation, a replicated array, or a large temporary. A match gives the investigation a place to start. The CLI can abstain when it recognizes neither an exception nor a symptom. [Diagnostic catalog](../diagnostics.md) · [Python API](../python-api.md)

### A concrete failure and its repair

Install with `python -m pip install jax-diagnostics`. With JAX already installed, put this example in `train.py` and run it.

```python
import jax
from jax_diagnostics import annotate_errors


@jax.jit
def choose(flag, x):
    if flag:
        return x + 1
    return x - 1


with annotate_errors():
    choose(True, 3)
```

Python needs a concrete boolean to execute the branch, but tracing represents the input with a placeholder value. The resulting `TracerBoolConversionError` gets a `JAX-TRACE-003` note in the traceback. The [repair guide](../reference/tracing.md#jax-trace-003) leads to a decision. For a data-dependent branch, express the branch as JAX control flow. For a genuine configuration parameter, static specialization can be appropriate, with a separate compilation for different static values. [JAX error reference](https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError)

Here the branch should depend on the runtime input. The fixed example verifies both outcomes using the same JIT-compiled function.

```python
import jax


@jax.jit
def choose(flag, x):
    return jax.lax.cond(flag, lambda value: value + 1,
                        lambda value: value - 1, x)


assert int(choose(True, 3)) == 4
assert int(choose(False, 3)) == 2
```

For a completed job, `jax-diagnose --json < traceback.txt` produces a route an agent can consume. To use the guides directly in a coding agent, run `npx skills add mlnomadpy/jax-diagnostics`. Give the agent the original failure and the expected behavior so it has a way to decide whether the change worked. [Quickstart](../quickstart.md) · [CLI and skill setup](../cli-and-skill.md)

## What we still need to prove

The current repository tests routing and exception annotation. We ran a [first paired benchmark](jax-repair-benchmark.md) with four reports and twelve trials. For this small, one-attempt study, all three conditions fixed all four cases. A larger evaluation should run the same model on the same bugs with a raw traceback, then with access to upstream docs, and finally with a traceback routed to a focused repair guide. Use independently written acceptance checks and include cases where the best response is to ask for more evidence. Track route accuracy separately from verified fix rate so a compelling explanation can't substitute for a passing reproduction.

The [roadmap](../roadmap.md) starts with real failure fixtures, better handling for chained exceptions and distributed logs, and narrower routes for Flax, Orbax, and Pallas where the evidence supports them. A useful contribution is a redacted traceback with the package versions, backend, smallest reproducer, and one check that distinguishes the fix from a workaround. [Report a diagnostic gap](https://github.com/mlnomadpy/jax-diagnostics/issues)
