PY=/opt/homebrew/bin/python3.11
OMP_NUM_THREADS=1 MOSEK_PAR_NUMTHREADS=1 JULIA_NUM_THREADS=1 \
  caffeinate -i -s "$PY" -u qi_from_psinteg_cth1.py 1         --force --no-plot > qi_cth1_full.log 2>&1 &
OMP_NUM_THREADS=1 MOSEK_PAR_NUMTHREADS=1 JULIA_NUM_THREADS=1 \
  caffeinate -i -s "$PY" -u qi_from_psinteg_cth1.py cut0.75_1 --force --no-plot > qi_cth1_cut.log  2>&1 &
wait
# Then plot once both are done
$PY qi_from_psinteg_cth1.py 1 cut0.75_1
