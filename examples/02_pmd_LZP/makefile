.phony: clean veryclean run test init
SHELL = /bin/bash

clean:
	rm -f in.vars.optzer.[1-9]* *~

veryclean: clean
	rm -rf out.cs.* subdir_*

init: veryclean
	rm -rf db.optzer.json out.optzer in.vars.optzer.best

run:
	optzer --nproc 5 --random-seed 42 2>&1 | tee out.optzer

test: init
	@echo " Testing optzer in examples/02_pmd_LZP; this could take a few minutes..."
	@optzer --nproc 5 --random-seed 42 2>&1 > out.optzer
	@python test.py 2>&1 > /dev/null || (echo " FAIL: examples/02_pmd_LZP" && exit 1)
	@echo " PASS: examples/02_pmd_LZP"

resdir := result_$(shell date "+%y%m%d_%H%M")
keep:
	mkdir -p $(resdir)
	cp -R in.* out.* data.* best_data $(resdir)/
