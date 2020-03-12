parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
(cd ${parent_path}/src/multiple-linear-regression; py main.py);