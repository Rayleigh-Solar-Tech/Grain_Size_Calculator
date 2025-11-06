"""
Runtime hook for NumPy to ensure proper initialization and avoid circular imports.
This runs before the main application starts.
"""

import sys
import os

# Ensure proper module loading order to prevent circular imports
if hasattr(sys, '_MEIPASS'):
    # We're running in a PyInstaller bundle
    
    # Set environment variables to help NumPy and prevent conflicts
    os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMPY_EXPERIMENTAL_DTYPE_API'] = '1'
    
    # Pre-import typing to prevent circular import issues
    try:
        import typing
        import types
        import collections.abc
    except ImportError:
        pass
    
    # Clear any cached NumPy typing modules that might cause conflicts
    modules_to_clear = [m for m in sys.modules.keys() if 'numpy.typing' in m]
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    # Ensure NumPy path is set correctly
    numpy_path = os.path.join(sys._MEIPASS, 'numpy')
    if os.path.exists(numpy_path):
        # Add NumPy path to the front of sys.path
        if numpy_path not in sys.path:
            sys.path.insert(0, numpy_path)