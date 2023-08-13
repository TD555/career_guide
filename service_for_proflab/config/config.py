import os

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
