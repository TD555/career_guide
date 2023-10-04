from waitress import serve
import sys
sys.path.insert(0, 'main')

from main.service import app, scheduler, job
from config.config import Config

if __name__ == "__main__":
    # hour = int(Config.JOB_HOUR)
    # if hour >= 4 and hour < 24:
    #     hour -=4
    # elif hour >= 0 and hour < 4: hour +=20
    # else: hour = 0
    
    # scheduler.add_job(id ='job',func=job, trigger='cron', hour=hour, minute=int(Config.JOB_MINUTE))
    # scheduler.start()
    # serve(app, port=5000)
    
    app.run(debug=True)