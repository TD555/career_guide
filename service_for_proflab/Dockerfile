FROM python:3.10.0

ARG DB_USER
ARG DB_PASSWORD
ARG DB_HOST
ARG DB_NAME
ARG DB_PORT


ENV DB_USER=$DB_USER
ENV DB_PASSWORD=$DB_PASSWORD
ENV DB_HOST=$DB_HOST
ENV DB_NAME=$DB_NAME
ENV DB_PORT=$DB_PORT


RUN apt-get update && apt-get install -y libpq-dev

RUN adduser --disabled-login service-proflab
USER service-proflab

WORKDIR  /var/www/service_proflab/service-for-proflab 

COPY --chown=service-proflab:service-proflab .   /var/www/service_proflab/service-for-proflab 

ENV PATH="$PATH:/home/service-proflab/.local/bin"

COPY --chown=service-proflab:service-proflab ./requirements.txt  /var/www/service_proflab/service-for-proflab/requirements.txt
RUN python3 -m pip install --no-cache-dir  -r requirements.txt && python3 -m spacy download en_core_web_md

CMD ["python3", "-u", "run.py"]