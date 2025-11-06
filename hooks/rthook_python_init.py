"""
Runtime hook to ensure proper Python interpreter initialization.
This fixes the "Failed to start embedded python interpreter" error.
"""

import sys
import os

# Set Python optimization flags
sys.dont_write_bytecode = True

# Ensure proper DLL loading for Windows
if hasattr(sys, '_MEIPASS') and os.name == 'nt':
    # Add the bundle directory to DLL search path
    import ctypes
    import ctypes.wintypes
    
    # Add current directory to DLL search path
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.SetDefaultDllDirectories(0x00001000)  # LOAD_LIBRARY_SEARCH_SYSTEM32
        kernel32.AddDllDirectory(sys._MEIPASS)
    except Exception:
        pass  # Ignore if this fails on older Windows versions
    
    # Set environment variables for proper Python initialization
    os.environ['PYTHONHOME'] = sys._MEIPASS
    os.environ['PYTHONPATH'] = sys._MEIPASS