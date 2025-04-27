.PHONY: venv install

VENV=.my_venv
PYTHON_VERSION=python3# Set or change python version if desired.
PYTHON=$(VENV)/bin/$(PYTHON_VERSION)

venv: # Create Virtual Environment
	$(PYTHON_VERSION) -m venv $(VENV)

install: # Install Dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --no-cache-dir -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name=$(VENV)