# ------- Development configuration
FROM python:3.9 as python-project-development
RUN apt-get update && apt-get install -y python3-tk python3-opencv gdal-bin libgdal-dev
COPY requirements_linux.txt requirements_linux.txt
RUN pip install --upgrade PIP
RUN pip install -r requirements_linux.txt

# ------- Run configuration
FROM python-project-development as python-project-application
WORKDIR /opt/otc/otvision
COPY . .
CMD python view.py
