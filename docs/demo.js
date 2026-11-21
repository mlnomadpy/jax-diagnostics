const guideBase = 'https://github.com/mlnomadpy/jax-diagnostics/blob/main/';
const examples = {
  tracer: {
    error: 'jax.errors.TracerBoolConversionError:\nAttempted boolean conversion of traced array.',
    id: 'JAX-TRACE-003',
    title: 'Traced Python boolean conversion',
    guide: 'references/tracing.md#jax-trace-003',
  },
  memory: {
    error: 'jax.errors.JaxRuntimeError:\nRESOURCE_EXHAUSTED: out of memory',
    id: 'JAX-MEM-001',
    title: 'Device memory exhaustion',
    guide: 'references/runtime.md#jax-mem-001',
  },
  rng: {
    error: 'jax.errors.KeyReuseError:\nPreviously consumed key passed to a JIT-compiled function.',
    id: 'JAX-RNG-001',
    title: 'PRNG key reused',
    guide: 'references/transforms.md#jax-rng-001',
  },
};

const buttons = document.querySelectorAll('button[data-example]');
for (const button of buttons) {
  button.addEventListener('click', () => {
    const example = examples[button.dataset.example];
    document.querySelector('#error').textContent = example.error;
    document.querySelector('#diagnostic-id').textContent = example.id;
    document.querySelector('#diagnostic-title').textContent = example.title;
    document.querySelector('#repair-guide').href = guideBase + example.guide;
    for (const item of buttons) {
      item.setAttribute('aria-pressed', String(item === button));
    }
  });
}
