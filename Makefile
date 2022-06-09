venv:
	python3 -m venv .venv

.ONESHELL:
install_m1: # install for Apple M1 machines
	brew install gdal 
	pip install fiona
	brew install proj
	pip install pyproj
	pip install pygeos
	pip install geopandas
	brew install python-tk
	pip install -r requirements_m1.txt

clean:
	rm -rf .venv


