FROM python:3.11-rc-slim as build
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc
RUN apt-get install -y --no-install-recommends build-essential gcc libpq-dev

WORKDIR /usr/app
RUN python -m venv /usr/app/venv
ENV PATH="/usr/app/venv/bin:$PATH"

COPY docker/requirements.txt requirements.txt
RUN pip install -U wheel pip
RUN pip install -r requirements.txt

FROM python:3.11-rc-slim@sha256:9a89111ec446d25d96e3503bb85c0aa12a6addaca61e441b0d801606afa86760
RUN apt-get update
RUN apt-get install -y nginx
RUN groupadd -g 999 python && useradd -r -u 999 -g python python
RUN mkdir /usr/app && chown python:python /usr/app
WORKDIR /usr/app
COPY --chown=python:python --from=build /usr/app/venv ./venv/
COPY --chown=python:python manage.py .
COPY --chown=python:python prediction/* ./prediction/
COPY --chown=python:python machinelearning/* ./machinelearning/
COPY --chown=python:python prediction/ ./prediction/
COPY --chown=python:python Dataset/ ./Dataset/
COPY --chown=python:python machinelearning/ ./machinelearning/
COPY --chown=python:python docker/settings.py machinelearning/settings.py
# COPY --chown=python:python docker/urls.py machinelearning/urls.py

COPY --chown=python:python docker/nginx.txt /etc/nginx/sites-available/default
RUN nginx

USER python

ENV PATH="/usr/app/venv/bin:$PATH"

RUN . ./venv/bin/activate
RUN python manage.py collectstatic --no-input
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "machinelearning.wsgi:application", "--log-level=debug", "--timeout=180", "--workers=3"]
