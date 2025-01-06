import os
import subprocess

# Define the folder to clone repositories
target_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dpu_packages'))
print(f"Cloning repositories in {target_folder}")
# Check whether conda or mamba is installed
conda_cmd = "conda"
if subprocess.run(["mamba", "--version"]).returncode != 0:
    conda_cmd = "mamba"
print(f"Using {conda_cmd} to install requirements")

os.makedirs(target_folder, exist_ok=True)

# List Git repositories
git_repos = [
    "https://github.com/spinoza-centre/prfpy_csenf.git",
    "https://github.com/mdaghlian/pycortex.git",
    "https://github.com/mdaghlian/figure_finder.git",
    "https://github.com/mdaghlian/prfpy_bayes.git"
]

# Clone and install each repository in editable mode
for repo in git_repos:
    repo_name = os.path.basename(repo).replace('.git', '')
    repo_path = os.path.join(target_folder, repo_name)
    
    # Clone the repository if not already cloned
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo, repo_path])        
    else: # Update the repository if already cloned
        subprocess.run(["git", "-C", repo_path, "pull"])
    
    # Try installing the requirements with mamba
    subprocess.run([conda_cmd, "install", "-y", "-c", "conda-forge", "-c", "defaults", "--file", os.path.join(repo_path, "requirements.txt")])
    subprocess.run(["pip", "install", "-r", os.path.join(repo_path, "requirements.txt")])
    
    # Install the repository in editable mode
    subprocess.run(["pip", "install", "-e", repo_path])    
