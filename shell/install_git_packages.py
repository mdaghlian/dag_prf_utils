import os
import subprocess

# Define the folder to clone repositories
target_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dpu_packages'))
print(f"Cloning repositories in {target_folder}")

os.makedirs(target_folder, exist_ok=True)

# List Git repositories
git_repos = [
    "https://github.com/spinoza-centre/prfpy_csenf.git",
    "https://github.com/mdaghlian/pycortex.git",
    "https://github.com/mdaghlian/figure_finder.git"
]

# Clone and install each repository in editable mode
for repo in git_repos:
    repo_name = os.path.basename(repo).replace('.git', '')
    repo_path = os.path.join(target_folder, repo_name)
    
    # Clone the repository if not already cloned
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo, repo_path])

    # Install the repository in editable mode
    subprocess.run(["pip", "install", "-e", repo_path])
