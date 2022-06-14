VENV = .venv
PY_VERSION = 3.9
PYTHON = $(VENV)/bin/python$(PY_VERSION)
PIP = $(VENV)/bin/pip
UNAME_S := $(shell uname -s)

run: install 
	$(PYTHON) view.py

install: $(VENV)/bin/activate 

$(VENV)/bin/activate: requirements_m1.txt
	python$(PY_VERSION) -m venv $(VENV); \
	if [ $(UNAME_S) = Linux ]; then \
		sudo apt-get install python3-tk; \
		$(PIP) install -r requirements_linux.txt; \
	fi ; \
	if [ $(UNAME_S) = Darwin ]; then \
		if [ $(shell uname -m) = arm64 ]; then \
			brew install gdal; \
			brew install proj; \
			brew install python-tk@$(PY_VERSION); \
			$(PIP) install -r requirements_m1.txt; \
		fi ; \
	fi 

test: requirements_dev.txt 
	$(PYTHON) -m pytest .

lint: requirements_dev.txt 
	$(PYTHON) -m flake8 OTVision tests
	$(PYTHON) -m yamllint .

format: requirements_dev.txt 
	$(PYTHON) -m isort .
	$(PYTHON) -m black .

dev: requirements_dev.txt 
	python$(PY_VERSION) -m venv $(VENV)
	$(PIP) install -e .
	$(PIP) install -r requirements_dev.txt
		
clean:
	rm -rf __pycache__
	rm -rf $(VENV)

.PHONY: run clean install test lint format


