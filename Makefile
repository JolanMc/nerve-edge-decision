# One command reproduces every number in results/: `make all` (about 20 min on 2 CPU cores).
PY ?= python
WORK ?= runs/v0.1
RAW ?= data

.PHONY: install all run report test lint quick demo clean

install:
	$(PY) -m pip install -e ".[train,dev]"

all: run report

run:
	$(PY) -m nerve_edge.pipeline --work $(WORK) --raw $(RAW)

report:
	$(PY) -m nerve_edge.report --work $(WORK) --out results

quick:
	$(PY) -m nerve_edge.pipeline --work runs/quick --raw $(RAW) --quick
	$(PY) -m nerve_edge.report --work runs/quick --out runs/quick/report

demo:
	$(PY) -m nerve_edge.demo --raw $(RAW)

test:
	$(PY) -m pytest -q

lint:
	ruff check .

clean:
	rm -rf runs/quick
