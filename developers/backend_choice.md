# Multiple Backend Architecture for `falwa`

## Common Pattern: Backend Registry + Dispatcher

The standard approach used by scientific Python packages (e.g., SciPy, scikit-learn, Matplotlib) is:

### 1. Directory Structure

```
src/falwa/
    __init__.py
    _backend.py            # backend selector (registry + get/set)
    backends/
        __init__.py
        _numba/            # pure-Python, always available
            __init__.py
            compute_qgpv.py
            ...
        _f2py/             # compiled, optional (may not be installed)
            __init__.py    # imports from the compiled .so modules
            ...
    oopinterface.py        # high-level code calls falwa._backend.get_backend()
    ...
```

### 2. The Backend Selector (`_backend.py`)

```python
_CURRENT_BACKEND = None

def set_backend(name: str):
    """User calls falwa.set_backend('numba') or falwa.set_backend('f2py')."""
    global _CURRENT_BACKEND
    if name == "f2py":
        from falwa.backends import _f2py as mod
    elif name == "numba":
        from falwa.backends import _numba as mod
    else:
        raise ValueError(f"Unknown backend: {name}")
    _CURRENT_BACKEND = mod

def get_backend():
    global _CURRENT_BACKEND
    if _CURRENT_BACKEND is None:
        # Auto-detect: prefer f2py if available, fall back to numba
        try:
            from falwa.backends import _f2py as mod
        except ImportError:
            from falwa.backends import _numba as mod
        _CURRENT_BACKEND = mod
    return _CURRENT_BACKEND
```

### 3. How Users Choose

Three common mechanisms (not mutually exclusive):

| Method | Example | When it's evaluated |
|---|---|---|
| **API call** | `falwa.set_backend("numba")` | Runtime, explicit |
| **Environment variable** | `FALWA_BACKEND=numba` | Startup, read in `_backend.py` |
| **Install-time extra** | `pip install falwa[f2py]` | Install time |

The **environment variable** is read as the default in `_backend.py`:

```python
import os
_DEFAULT = os.environ.get("FALWA_BACKEND", "auto")  # "auto", "numba", or "f2py"
```

### 4. How High-Level Code Dispatches

```python
# In oopinterface.py
from falwa._backend import get_backend

class QGFieldBase:
    def _compute_qgpv(self, ...):
        backend = get_backend()
        return backend.compute_qgpv(...)   # same function signature in both backends
```

The key contract: **every backend module exposes the same public functions with identical signatures**.

### 5. Build System (`pyproject.toml`)

Since F2PY requires a Fortran compiler + Meson while Numba is pure Python, use **optional dependencies**:

```toml
[build-system]
requires = ["setuptools>=64", "numpy"]
build-backend = "setuptools.build_meta"

[project]
dependencies = ["numpy", "scipy", "numba"]  # numba backend always works

[project.optional-dependencies]
f2py = []  # marker only — user must build from source with meson for this
```

However, the F2PY backend complicates things because it needs a *different build system* (meson-python). Two common solutions:

- **Separate package**: Ship the Fortran extensions as `falwa-f2py` (a separate PyPI package that installs into `falwa.backends._f2py`). The main `falwa` package auto-detects it.
- **Single repo, conditional build**: Use meson-python as the build backend and make the Numba backend the fallback when the compiled extensions aren't available.

### Summary

| Concern | Solution |
|---|---|
| Where backends live | `falwa/backends/_numba/`, `falwa/backends/_f2py/` |
| How user selects | `set_backend()` API + `FALWA_BACKEND` env var |
| How code dispatches | `get_backend().function_name(...)` |
| Fallback | Auto-detect: try F2PY import → fall back to Numba |
| Build system | Numba is always available; F2PY is optional (separate package or conditional build) |

The **auto-detect with fallback** pattern is the most user-friendly — the package always works (via Numba), and users who install the Fortran extensions get them automatically.

