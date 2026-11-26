SHELL := /bin/sh
PY ?= python3

.PHONY: help dry-run reproduce langgraph clean

help:
	@echo "make dry-run     plumbing check, no API key"
	@echo "make reproduce   real evidence run -> results/"
	@echo "make langgraph   LangGraph adapter run -> results/"

dry-run:
	$(PY) experiments/fork-fidelity/fork_fidelity_suite.py --dry-run --report-dir results

reproduce:
	$(PY) experiments/fork-fidelity/fork_fidelity_suite.py \
		--repeats 3 --judge-reps 3 --report-dir results

langgraph:
	$(PY) experiments/fork-fidelity/fork_fidelity_langgraph.py \
		--report-dir results

clean:
	rm -f results/*.tmp
