FROM python:3.8

COPY requirements.txt /
RUN set -ex && \
    pip install -r /requirements.txt
EXPOSE 8080
COPY ./app /app

WORKDIR /app

CMD exec python incident_dashboard.py
