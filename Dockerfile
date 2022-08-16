FROM python:3.11-rc-slim as build
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

WORKDIR /usr/app
RUN python -m venv /usr/app/venv
ENV PATH="/usr/app/venv/bin:$PATH"

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt

FROM python:3.11-rc-slim@sha256:9a89111ec446d25d96e3503bb85c0aa12a6addaca61e441b0d801606afa86760
RUN groupadd -g 999 python && useradd -r -u 999 -g python python
RUN mkdir /usr/app && chown python:python /usr/app
WORKDIR /usr/app
COPY --chown=python:python --from=build /usr/app/venv ./venv/
COPY --chown=python:python manage.py .
COPY --chown=python:python prediction/* ./prediction/
COPY --chown=python:python machinelearning/* ./machinelearning/
COPY --chown=python:python docker/settings.py machinelearning/settings.py

USER python

ENV PATH="/usr/app/venv/bin:$PATH"

RUN . ./venv/bin/activate
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "machinelearning.wsgi:application", "--log-level=debug"]
