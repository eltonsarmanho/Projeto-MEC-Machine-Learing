FROM python:3.10-slim as build
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

WORKDIR /usr/app
RUN python -m venv /usr/app/venv
ENV PATH="/usr/app/venv/bin:$PATH"

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt

FROM python:3.10-slim@sha256:cf85cd32e60184a94d88a0103c289d09024abffaa77680d116d7cc837668ea15
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
