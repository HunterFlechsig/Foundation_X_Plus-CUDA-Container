#!/usr/bin/env bash
set -euo pipefail

# Report every path the CANDID-PTX test run needs that is not on disk.
#   ./scripts-apptainer/check_dataset_paths.sh
#   ./scripts-apptainer/check_dataset_paths.sh config/dataset_locations_asu_sol.yml

exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_dataset_paths.py" "$@"
