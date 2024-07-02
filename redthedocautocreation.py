import platform

# Determine Python version
python_version = platform.python_version()

# Example list of system packages
system_packages = ['graphviz', 'make']

# Generate .readthedocs.yaml content
yaml_content = f"""
version: 2

build:
  os: {platform.system().lower()}  # Auto-detect OS
  tools:
    python: "{python_version}"  # Auto-detect Python version
  jobs:
    pre_build:
      - {' && '.join([f"apt-get install -y {pkg}" for pkg in system_packages])}  # Install system packages
    post_build:
      - echo "Build finished"
"""

# Write the generated content to .readthedocs.yaml file
with open('.readthedocs.yaml', 'w') as f:
    f.write(yaml_content)
