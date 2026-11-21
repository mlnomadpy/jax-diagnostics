"""Route a JAX error name, traceback, or symptom to a local repair guide."""

import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Diagnosis:
    id: str
    guide: str
    title: str
    evidence: str = ""
    match: str = ""

    @property
    def link(self) -> str:
        return f"references/{self.guide}.md#{self.id.lower()}"


# Names exported by jax.errors, with one specific guide section per class.
ERRORS = {
    "ConcretizationTypeError": Diagnosis("JAX-TRACE-001", "tracing", "Concrete value required during tracing"),
    "UnexpectedTracerError": Diagnosis("JAX-TRACE-002", "tracing", "Tracer escaped a transform"),
    "TracerBoolConversionError": Diagnosis("JAX-TRACE-003", "tracing", "Traced Python boolean conversion"),
    "TracerIntegerConversionError": Diagnosis("JAX-TRACE-004", "tracing", "Traced Python integer conversion"),
    "TracerArrayConversionError": Diagnosis("JAX-TRACE-005", "tracing", "Traced NumPy array conversion"),
    "NonConcreteBooleanIndexError": Diagnosis("JAX-TRACE-006", "tracing", "Boolean indexing needs a static result shape"),
    "InconclusiveDimensionOperation": Diagnosis("JAX-SHAPE-001", "shapes", "Symbolic dimension comparison cannot be proved"),
    "JaxprTypeError": Diagnosis("JAX-TRANSFORM-001", "transforms", "Invalid Jaxpr type or transform rule"),
    "JAXTypeError": Diagnosis("JAX-TYPE-001", "shapes", "JAX type error"),
    "JAXIndexError": Diagnosis("JAX-INDEX-001", "shapes", "JAX indexing error"),
    "KeyReuseError": Diagnosis("JAX-RNG-001", "transforms", "PRNG key reused"),
    "JaxRuntimeError": Diagnosis("JAX-RUNTIME-001", "runtime", "JAX backend runtime error"),
}

# More specific symptoms win when multiple phrases describe the same failure.
SYMPTOMS = (
    (r"mosaic|\bpallas\b|semaphore deadlock|\bvmem\b", Diagnosis("JAX-KERNEL-001", "runtime", "Pallas or Mosaic kernel failure")),
    (r"RESOURCE_EXHAUSTED|out of memory|\boom\b", Diagnosis("JAX-MEM-001", "runtime", "Device memory exhaustion")),
    (r"non[- ]finite|\bnan(?:s)?\b|\binf(?:s)?\b|\binfinity\b", Diagnosis("JAX-NUM-001", "numerics", "Non-finite numerical result")),
    (r"\borbax\b|checkpoint (?:restore|mismatch)|restore.{0,50}checkpoint", Diagnosis("JAX-RESTORE-001", "runtime", "Checkpoint restore or resume mismatch")),
    (r"collective.{0,40}(?:hang|timeout|timed out)|rendezvous|coordinator.{0,40}(?:timeout|timed out)|distributed.{0,40}(?:hang|timeout)", Diagnosis("JAX-DIST-001", "runtime", "Distributed initialization or collective failure")),
    (r"\bsharding\b|\bPartitionSpec\b|unexpected replication", Diagnosis("JAX-SHARD-001", "runtime", "Sharding or mesh mismatch")),
    (r"transfer guard|disallowed (?:host|device)[- ]to[- ](?:host|device) transfer", Diagnosis("JAX-TRANSFER-001", "runtime", "Disallowed host or device transfer")),
    (r"buffer.{0,30}(?:deleted|donated)|donated.{0,30}(?:buffer|argument)|Array has been deleted", Diagnosis("JAX-BUFFER-001", "runtime", "Buffer reused after donation or deletion")),
    (r"unable to initialize backend|no (?:gpu|tpu) devices|\blibtpu\b|\bcuDNN\b|CUDA_ERROR|jaxlib.{0,45}version|PJRT plugin", Diagnosis("JAX-BACKEND-001", "runtime", "Backend or runtime initialization failure")),
    (r"recompil|cache miss|slow compil|slow tracing", Diagnosis("JAX-COMPILE-001", "runtime", "Tracing or compilation overhead")),
    (r"\bnnx\b|ScopeParamShapeError|ModifyScopeVariableError|ScopeVariableNotFoundError|InvalidRngError", Diagnosis("JAX-STATE-001", "transforms", "Flax state or transform mismatch")),
    (r"\bparity\b|pytorch.{0,30}(?:drift|mismatch)|framework.{0,25}mismatch", Diagnosis("JAX-PARITY-001", "numerics", "Cross-framework numerical mismatch")),
    (r"Gradient only defined for scalar|grad requires real|custom_(?:jvp|vjp)|gradient.{0,25}(?:zero|missing|incorrect)", Diagnosis("JAX-GRAD-001", "transforms", "Gradient or custom derivative failure")),
    (r"inconsistent sizes for array axes|scan.{0,45}carry|mapped axis|vmap.{0,30}in_axes", Diagnosis("JAX-AXIS-001", "transforms", "Mapped axis or scan carry mismatch")),
    (r"broadcast|rank promotion|shape mismatch|dot_general requires contracting", Diagnosis("JAX-SHAPE-002", "shapes", "Shape, rank, or broadcasting mismatch")),
    (r"dtype|type promotion|x64", Diagnosis("JAX-TYPE-001", "shapes", "JAX type error")),
)

HEADER = re.compile(
    r"^[ \t]*(?:\|[ \t]*)?(?:[\w]+\.)*(?P<name>[\w]*(?:Error|Exception)|"
    r"InconclusiveDimensionOperation)(?::[ \t]*.*)?[ \t]*$", re.MULTILINE,
)
GENERIC = {"JaxRuntimeError", "JAXTypeError", "JAXIndexError"}


def diagnose(source: str) -> Diagnosis | None:
    """Use exception headers ahead of symptom text, without evaluating input."""
    source = source.strip()
    headers = list(HEADER.finditer(source))
    for header in reversed(headers):
        name = header["name"]
        if name in ERRORS and name not in GENERIC:
            return replace(ERRORS[name], evidence=name, match="exception")
        if name in GENERIC:
            break
    message = source[headers[-1].start():] if headers else source
    for pattern, route in SYMPTOMS:
        matched = re.search(pattern, message, re.IGNORECASE)
        if matched:
            return replace(route, evidence=matched[0], match="symptom")
    for header in reversed(headers):
        name = header["name"]
        if name in ERRORS:
            return replace(ERRORS[name], evidence=name, match="exception")
    return None
