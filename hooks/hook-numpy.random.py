#!/usr/bin/env python3
"""
Hook to handle numpy.random import issues in PyInstaller.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all numpy.random submodules
hiddenimports = collect_submodules('numpy.random')

# Exclude problematic modules that cause circular imports
excludedimports = ['numpy.random._pickle']

# Add specific modules that are needed
hiddenimports += [
    'numpy.random.bit_generator',
    'numpy.random._common',
    'numpy.random._generator',
    'numpy.random.mtrand',
]

# Collect data files
datas = collect_data_files('numpy.random')