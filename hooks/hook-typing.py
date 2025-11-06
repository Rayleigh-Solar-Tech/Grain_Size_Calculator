"""
Custom PyInstaller hook to fix typing module conflicts with NumPy.
This prevents circular import issues between typing and numpy.typing modules.
"""

# Exclude problematic typing modules that conflict with NumPy
excludedimports = [
    'numpy.typing._generic_alias',
    'numpy.typing._shape', 
    'numpy.typing._scalars',
    'numpy.typing._array_like',
    'numpy.typing._dtype_like',
    'numpy.typing._ufunc',
    'numpy.typing.mypy_plugin',
]

# Include only essential typing functionality
hiddenimports = [
    'typing',
    'types',
    'collections.abc',
]