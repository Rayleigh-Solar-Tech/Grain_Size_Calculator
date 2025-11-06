"""
Runtime hook to monkey-patch numpy imports and avoid test modules.
This must run before numpy is imported.
"""
import sys
import importlib.util
import types

# Store the original import function
original_import = __builtins__.__import__

def create_mock_pytesttester():
    """Create a mock module for numpy._pytesttester with expected classes."""
    mock_module = types.ModuleType('numpy._pytesttester')
    
    # Create a mock PytestTester class
    class MockPytestTester:
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, *args, **kwargs):
            return True
        
        def test(self, *args, **kwargs):
            return True
    
    mock_module.PytestTester = MockPytestTester
    
    # Add any other commonly used attributes
    def dummy_function(*args, **kwargs):
        return True
    
    # Dynamically handle any attribute access
    def __getattr__(name):
        print(f"NumPy patch: Mocking access to _pytesttester.{name}")
        return dummy_function
    
    mock_module.__getattr__ = __getattr__
    return mock_module

def create_mock_multiarray_tests():
    """Create a mock module for numpy._core._multiarray_tests with expected attributes."""
    mock_module = types.ModuleType('numpy._core._multiarray_tests')
    
    # Add common attributes that numpy might expect
    mock_module.format_float_OSprintf_g = lambda *args, **kwargs: ""
    mock_module.format_float_OSprintf_f = lambda *args, **kwargs: ""
    mock_module.format_float_OSprintf_e = lambda *args, **kwargs: ""
    mock_module.format_float_OSprintf_F = lambda *args, **kwargs: ""
    mock_module.format_float_OSprintf_E = lambda *args, **kwargs: ""
    mock_module.format_float_OSprintf_G = lambda *args, **kwargs: ""
    
    # Add any other commonly used attributes as no-ops
    def dummy_function(*args, **kwargs):
        return None
    
    # Dynamically handle any attribute access
    def __getattr__(name):
        print(f"NumPy patch: Mocking access to _multiarray_tests.{name}")
        return dummy_function
    
    mock_module.__getattr__ = __getattr__
    return mock_module

def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Patched import function that skips problematic numpy test modules.
    """
    # Check if we're trying to import a problematic numpy test module
    if '_multiarray_tests' in name or '_pytesttester' in name or (name.startswith('numpy') and 'test' in name):
        print(f"NumPy patch: Blocking import of test module: {name}")
        
        # For the specific _multiarray_tests module, return our mock
        if '_multiarray_tests' in name:
            mock_module = create_mock_multiarray_tests()
            sys.modules[name] = mock_module
            return mock_module
        elif '_pytesttester' in name:
            mock_module = create_mock_pytesttester()
            sys.modules[name] = mock_module
            return mock_module
        else:
            # For other test modules, return a simple dummy
            spec = importlib.util.spec_from_loader(name, loader=None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            return module
    
    # Use the original import for everything else
    return original_import(name, globals, locals, fromlist, level)

# Monkey-patch the import function
__builtins__.__import__ = patched_import

print("NumPy runtime hook: Installed import patcher to block test modules")