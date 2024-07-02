import subprocess
import pkg_resources

def get_installed_packages():
    # Get a list of installed packages
    installed_packages = pkg_resources.working_set
    installed_packages_dict = {pkg.key: pkg.version for pkg in installed_packages}
    return installed_packages_dict

def get_requirements(file_path):
    # Read and parse the requirements.txt file
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    requirements_dict = {}
    for req in requirements:
        if "==" in req:
            pkg, version = req.strip().split("==")
            requirements_dict[pkg] = version
        else:
            requirements_dict[req.strip()] = None
    return requirements_dict

def compare_versions(installed, required):
    differences = {"missing": [], "different_versions": {}}
    
    for pkg, req_version in required.items():
        if pkg not in installed:
            differences["missing"].append(pkg)
        elif req_version and installed[pkg] != req_version:
            differences["different_versions"][pkg] = {
                "installed": installed[pkg],
                "required": req_version
            }
    
    return differences

def main():
    requirements_file = 'requirements.txt'  # Adjust the path if necessary
    installed_packages = get_installed_packages()
    required_packages = get_requirements(requirements_file)
    
    differences = compare_versions(installed_packages, required_packages)
    
    print("Missing packages:")
    for pkg in differences["missing"]:
        print(f"- {pkg}")
    
    print("\nPackages with different versions:")
    for pkg, versions in differences["different_versions"].items():
        print(f"- {pkg}: installed {versions['installed']}, required {versions['required']}")

if __name__ == "__main__":
    main()
