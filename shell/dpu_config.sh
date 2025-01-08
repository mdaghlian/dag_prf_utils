# specify the folder where dag_prf_utils is held
DPU_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPU_SETUP_FILE="${DPU_SETUP_DIR}/dpu_setup"
DPU_REPO_DIR=$(dirname ${DPU_SETUP_DIR})
export DAG_UTILS=$DPU_REPO_DIR

# Configurartion for DPU setup
install_git_files="True" # True or False
install_cmd="mamba install -y --file " # or "conda install -y --file " "mamba install -y --file " "pip install -r "