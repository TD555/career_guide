FROM python:3.10.0

# Set environment variables for database connection
ENV DB_USER=proflab_user
ENV DB_PASSWORD=puacX9KK[@mX
ENV DB_HOST=localhost
ENV DB_NAME=proflab_db
ENV DB_PORT=5432

# Install PostgreSQL client library
RUN apt-get update && apt-get install -y libpq-dev

RUN adduser --disabled-login service-proflab
USER service-proflab

WORKDIR  /var/www/service_proflab/service-for-proflab 

COPY --chown=service-proflab:service-proflab .   /var/www/service_proflab/service-for-proflab 

COPY --chown=service-proflab:service-proflab ./requirements.txt  /var/www/service_proflab/service-for-proflab/requirements.txt
RUN python3 -m pip install --no-cache-dir  -r requirements.txt && python3 -m spacy download en_core_web_md

CMD ["python3", "run.py"]
