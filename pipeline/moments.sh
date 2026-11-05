set -eu
PY=${PYTHON:-python3}
: "${DSEC_ROOT:?Set DSEC_ROOT to the extracted dataset directory}"
: "${AFTERDARK_MOMENTS_OUT:?Set AFTERDARK_MOMENTS_OUT to the output directory}"
OUT=$AFTERDARK_MOMENTS_OUT
SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
"$PY" "$SCRIPT_DIR/export_moment.py" --root "$DSEC_ROOT" --seq zurich_city_09_c --t0 58305557568 --seconds 9 --out "$OUT/crosswalk" &
"$PY" "$SCRIPT_DIR/export_moment.py" --root "$DSEC_ROOT" --seq zurich_city_10_b --t0 57466057236 --seconds 6 --out "$OUT/zone30" &
"$PY" "$SCRIPT_DIR/export_moment.py" --root "$DSEC_ROOT" --seq zurich_city_10_b --t0 57425507383 --seconds 6 --out "$OUT/street" &
"$PY" "$SCRIPT_DIR/export_moment.py" --root "$DSEC_ROOT" --seq zurich_city_09_e --t0 58516000000 --seconds 5 --out "$OUT/rider" &
wait
