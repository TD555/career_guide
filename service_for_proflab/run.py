from waitress import serve
import sys
sys.path.insert(0, 'main')

from main.service import app, scheduler, job
from config.config import Config

if __name__ == "__main__":
    scheduler.add_job(id ='job',func=job, trigger='cron', hour=20, minute=0)
    scheduler.start()
    serve(app, port=Config.APP_PORT)