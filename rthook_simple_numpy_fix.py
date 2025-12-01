#!/usr/bin/env python3
"""
Simple runtime hook to fix ONLY the numpy._core._multiarray_tests import error.
This is a minimal fix without complex import patching.
"""

import sys
import types

def create_mock_multiarray_tests():
    """Create a comprehensive mock for numpy._core._multiarray_tests."""
    mock_module = types.ModuleType('numpy._core._multiarray_tests')
    
    # Add the essential attributes and functions that NumPy expects
    mock_module.__file__ = '<mock>'
    mock_module.__path__ = []
    
    # Mock functions that are commonly expected
    def mock_function(*args, **kwargs):
        """Generic mock function that returns None or empty result."""
        return None
    
    # Add all the specific attributes that NumPy might look for
    mock_attributes = [
        'format_float_OSprintf_g',
        'format_float_positional', 
        'format_float_scientific',
        'test_neighborhood_iter',
        'test_neighborhood_iter_oob',
        'test_pydatamem_seteventhook_start',
        'test_pydatamem_seteventhook_end',
        'get_buffer_info',
        'get_ndarray_c_version',
        'get_ndarray_cfa_base',
        'test_neighborhood_iter_circular',
    ]
    
    # Set all expected attributes to mock functions
    for attr_name in mock_attributes:
        setattr(mock_module, attr_name, mock_function)
    
    # Add any other attributes that might be needed
    mock_module.__all__ = mock_attributes
    
    return mock_module

def create_mock_pytesttester():
    """Create a mock for numpy._core._pytesttester."""
    mock_module = types.ModuleType('numpy._core._pytesttester')
    
    # Add basic attributes
    mock_module.__file__ = '<mock>'
    mock_module.__path__ = []
    
    # Mock the PytestTester class that might be expected
    class MockPytestTester:
        def __init__(self, module_name=None):
            self.module_name = module_name
        
        def test(self, *args, **kwargs):
            return True
            
        def bench(self, *args, **kwargs):
            return True
    
    mock_module.PytestTester = MockPytestTester
    
    return mock_module

# Create and register the mock modules
try:
    # Mock numpy._core._multiarray_tests
    if 'numpy._core._multiarray_tests' not in sys.modules:
        sys.modules['numpy._core._multiarray_tests'] = create_mock_multiarray_tests()
        print("DEBUG: Created mock for numpy._core._multiarray_tests")
    
    # Mock numpy._core._pytesttester
    if 'numpy._core._pytesttester' not in sys.modules:
        sys.modules['numpy._core._pytesttester'] = create_mock_pytesttester()
        print("DEBUG: Created mock for numpy._core._pytesttester")
    
    # Also handle the old names for backward compatibility
    if 'numpy.core._multiarray_tests' not in sys.modules:
        sys.modules['numpy.core._multiarray_tests'] = sys.modules['numpy._core._multiarray_tests']
        print("DEBUG: Created alias for numpy.core._multiarray_tests")
    
    if 'numpy.core._pytesttester' not in sys.modules:
        sys.modules['numpy.core._pytesttester'] = sys.modules['numpy._core._pytesttester']
        print("DEBUG: Created alias for numpy.core._pytesttester")
        
except Exception as e:
    print(f"DEBUG: Error creating NumPy mocks: {e}")
    # Continue anyway - the application might still work