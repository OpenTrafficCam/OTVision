# ------- Development configuration
FROM python:3.9 as python-project-development
RUN apt-get update \
    && apt-get install --no-install-recommends -y python3-tk=3.9.2-1 \
    python3-opencv=4.5.1+dfsg-5 gdal-bin=3.2.2+dfsg-2+deb11u2 \
    libgdal-dev=3.2.2+dfsg-2+deb11u2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /platomo/ptvalidate
COPY requirements_linux.txt requirements_linux.txt
RUN pip install --no-cache-dir --upgrade pip==22.3 && pip install --no-cache-dir -r requirements_linux.txt
ENV PYTHONPATH /opt/otc/otvision

# ------- Run configuration
FROM python-project-development as python-project-application
WORKDIR /opt/otc/otvision
COPY . .
CMD ["python", "view.py"]
