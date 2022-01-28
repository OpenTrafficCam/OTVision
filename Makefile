venv:
	python3 -m venv .venv

install: _otv _install

install_m1: _otv _install_m1

install_dev_m1: _otv_dev install_m1

install_dev: _otv_dev _install

_install:
	pip install -r requirements_m1.txt

.ONESHELL:
_install_m1: # install for Apple M1 machines
	brew install gdal proj
	brew install python-tk
	pip install -r requirements_m1.txt

_otv_dev:
	pip install -e .

_otv:
	pip install -e .

clean:
	rm -rf .venv


