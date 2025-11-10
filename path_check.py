import sys
import os

print("\n=== ENVIRONMENT DIAGNOSTICS ===")
print(f"Project Path: {os.getcwd()}")
print(f"Python Executable: {sys.executable}")
print("\nSYS.PATH:")
for path in sys.path:
    print(f" - {path}")

print("\nMODULE PATHS:")
try:
    import transformers
    print(f"Transformers: {transformers.__file__}")
except ImportError:
    print("Transformers: NOT FOUND")

try:
    import evaluate
    print(f"Evaluate: {evaluate.__file__}")
except ImportError:
    print("Evaluate: NOT FOUND")

print("\nENVIRONMENT VARIABLES:")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"MODULEPATH: {os.environ.get('MODULEPATH', 'Not set')}")