#!/usr/bin/env python3
"""
Runtime hook to fix NumPy random module circular import issue in PyInstaller.
This fixes the 'SystemRandom' import error that occurs when numpy.random conflicts with built-in random.
"""

import sys
import importlib

# Fix for numpy.random circular import issue
def fix_numpy_random_import():
    """Fix circular import between numpy.random and built-in random module."""
    try:
        # Force import of built-in random module first
        import random as builtin_random
        
        # Ensure SystemRandom is available
        if not hasattr(builtin_random, 'SystemRandom'):
            from random import SystemRandom
            builtin_random.SystemRandom = SystemRandom
            
        # Now safely import numpy random
        import numpy.random
        
        print("DEBUG: NumPy random import fix applied successfully")
        
    except Exception as e:
        print(f"DEBUG: NumPy random import fix failed: {e}")
        # Continue anyway, might still work

# Apply the fix immediately when this hook runs
fix_numpy_random_import()