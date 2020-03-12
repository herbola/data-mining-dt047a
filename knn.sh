parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
(cd ${parent_path}/src/k-nearest-neighbors; py main.py);