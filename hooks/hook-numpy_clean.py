"""
Custom PyInstaller hook for numpy that completely excludes test modules.
This hook prevents test modules from being included in the build.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all numpy submodules except test modules
numpy_modules = collect_submodules('numpy')

# Filter out test modules
excluded_patterns = [
    '_multiarray_tests',
    'tests',
    'testing',
    'conftest',
    'test_',
    '.test_',
    '_test',
    '.tests.',
    'numpy.typing',  # Also exclude typing which causes issues
]

filtered_modules = []
for module in numpy_modules:
    should_exclude = False
    for pattern in excluded_patterns:
        if pattern in module:
            should_exclude = True
            break
    if not should_exclude:
        filtered_modules.append(module)

hiddenimports = filtered_modules

# Collect data files but exclude test data
datas = collect_data_files('numpy', excludes=['tests', 'test_*', '**/tests/**', '**/test_*'])

print(f"NumPy hook: Including {len(hiddenimports)} modules, excluding {len(numpy_modules) - len(hiddenimports)} test modules")