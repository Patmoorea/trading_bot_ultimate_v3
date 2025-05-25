import platform, sys, os
print(f"Python: {sys.version}")
print(f"VENV: {os.path.exists('.venv/bin/python')}")
print(f"Arch: {platform.machine()}")
