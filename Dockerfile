FROM python:3.9

COPY requirements.txt /
RUN set -ex && \
    pip install -r /requirements.txt
EXPOSE 8080

COPY ./app /app
WORKDIR /app

CMD exec gunicorn -b 0.0.0.0:8080  --worker-class gevent --threads 8 -t 100 tdot_dashboard_historical:server
