# src/h2o_init.py
import time
import h2o

h2o.init(
    ip="127.0.0.1",
    port=54321,
    nthreads=2,
    max_mem_size="4G"
)

h2o.cluster().show_status()


try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    pass
