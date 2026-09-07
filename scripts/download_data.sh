set -euo pipefail

COMP="nfl-health-and-safety-helmet-assignment"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
KAGGLE="$ROOT/.venv/bin/kaggle"
PY="$ROOT/.venv/bin/python"

mode="${1:-list}"
mkdir -p "$RAW"

unpack() {
  find "$RAW" -name '*.zip' -exec unzip -q -n -d "$RAW" {} \; -delete
}

case "$mode" in
  list)
    "$PY" "$ROOT/scripts/list_files.py"
    ;;
  tables)
    # list_files.py pages through the API; the CLI's own listing stops at 20.
    "$PY" "$ROOT/scripts/list_files.py" --suffix .csv --names-only | while read -r f; do
      echo ">> $f"
      "$KAGGLE" competitions download -c "$COMP" -f "$f" -p "$RAW"
    done
    unpack
    ;;
  all)
    "$KAGGLE" competitions download -c "$COMP" -p "$RAW"
    unpack
    ;;
  *)
    echo "usage: $0 {list|tables|all}" >&2
    exit 1
    ;;
esac

echo
echo "data/raw now holds:"
ls -lh "$RAW"
