"""
Runtime hook to fix NumPy import issues in PyInstaller bundle.
This hook patches numpy imports to avoid test modules.
"""
import sys
import os

# Remove any problematic modules from sys.modules if they exist
problematic_modules = [
    'numpy._core._multiarray_tests',
    'numpy.core._multiarray_tests',
    'numpy._core.tests',
    'numpy.core.tests',
    'numpy.testing',
    'numpy.conftest'
]

for module in problematic_modules:
    if module in sys.modules:
        del sys.modules[module]

# Patch numpy._core.function_base to avoid importing test modules
def patch_numpy_function_base():
    """Patch numpy to avoid importing test modules."""
    try:
        # Import numpy._core.function_base and patch the add_newdoc function
        import numpy._core.function_base as fb
        
        # Store the original add_newdoc function
        original_add_newdoc = fb.add_newdoc
        
        def patched_add_newdoc(place, obj, doc, warn_on_python=True):
            """Patched add_newdoc that avoids importing test modules."""
            try:
                # Check if we're trying to import test modules
                if isinstance(obj, str) and '_multiarray_tests' in obj:
                    print(f"NumPy patch: Skipping test module import: {obj}")
                    return
                return original_add_newdoc(place, obj, doc, warn_on_python)
            except Exception as e:
                print(f"NumPy patch warning: {e}")
                return
        
        # Replace the function
        fb.add_newdoc = patched_add_newdoc
        print("NumPy runtime hook: Successfully patched add_newdoc function")
        
    except Exception as e:
        print(f"NumPy runtime hook: Failed to patch function_base: {e}")

# Apply patches before numpy is fully imported
patch_numpy_function_base()

# Now try to import numpy properly
try:
    import numpy
    # Force import of essential numpy modules
    import numpy.core._multiarray_umath
    import numpy.core.multiarray
    import numpy.core.umath
    print("NumPy runtime hook: Successfully initialized NumPy core modules")
except ImportError as e:
    print(f"NumPy runtime hook warning: {e}")
    pass