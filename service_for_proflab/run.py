from waitress import serve
import sys
sys.path.insert(0, 'main')

from service import app
from config.config import Config
from waitress import serve

if __name__ == "__main__":
    # app.run(debug=True)
    serve(app, port=Config.APP_PORT)
