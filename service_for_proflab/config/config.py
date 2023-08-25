import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # PostgreSQL configurations
    DATABASE_USER = os.environ.get('DB_USER')
    DATABASE_PASSWORD = os.environ.get('DB_PASSWORD')
    DATABASE_HOST = os.environ.get('DB_HOST')
    DATABASE_PORT = os.environ.get('DB_PORT', 5432)
    DATABASE_NAME = os.environ.get('DB_NAME')

    # API key for OpanAI
    API_KEY = os.environ.get('API_KEY')
    
    # APP port
    APP_PORT = os.environ.get('APP_PORT')
    
    # Token validation
    API_DOCS = os.environ.get('API_DOCS')

    # Cron Job time
    JOB_HOUR = os.environ.get('JOB_HOUR')
    JOB_MINUTE = os.environ.get('JOB_MINUTE')