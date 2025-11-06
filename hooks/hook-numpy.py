"""
Custom PyInstaller hook for NumPy to fix import issues.
This hook ensures all necessary NumPy modules are included and fixes circular import issues.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, is_module_satisfies

# Collect all NumPy submodules but exclude problematic ones
hiddenimports = []

# Core NumPy modules
core_modules = [
    'numpy.core',
    'numpy._core',
    'numpy.lib',
    'numpy.linalg',
    'numpy.random',
    'numpy.fft',
    'numpy.polynomial',
    'numpy.ma',
    'numpy.testing',
]

for module in core_modules:
    try:
        hiddenimports.extend(collect_submodules(module))
    except Exception:
        pass

# Add specific modules that are often missed
hiddenimports += [
    'numpy._core._multiarray_tests',
    'numpy._core._multiarray_umath',
    'numpy.core._multiarray_tests',
    'numpy.core._multiarray_umath',
    'numpy.linalg._umath_linalg',
    'numpy.random._common',
    'numpy.random._generator',
    'numpy.random._mt19937',
    'numpy.random._pcg64',
    'numpy.random._philox',
    'numpy.random._sfc64',
    'numpy.random.bit_generator',
]

# Exclude problematic modules that cause circular imports
excludes = [
    'numpy.typing._generic_alias',
    'numpy.typing._shape',
    'numpy.typing._scalars',
    'numpy.typing._array_like',
    'numpy.typing._dtype_like',
    'numpy.typing._ufunc',
    'numpy.distutils',
    'numpy.f2py',
    'numpy.doc',
]

# Filter out excluded modules
hiddenimports = [mod for mod in hiddenimports if not any(mod.startswith(exc) for exc in excludes)]

# Collect NumPy data files
datas = collect_data_files('numpy', excludes=['**/*.pyc', '**/*.pyo'])