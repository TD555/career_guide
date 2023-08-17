FROM python:3.10.0

RUN adduser --disabled-login service-proflab
USER service-proflab

WORKDIR  /var/www/service_proflab/service-for-proflab 

COPY --chown=service-proflab:service-proflab .   /var/www/service_proflab/service-for-proflab 

COPY --chown=service-proflab:service-proflab ./requirements.txt  /var/www/service_proflab/service-for-proflab/requirements.txt
RUN python3 -m pip install --no-cache-dir  -r requirements.txt && python3 -m spacy download en_core_web_md

CMD ["python3", "run.py"]
