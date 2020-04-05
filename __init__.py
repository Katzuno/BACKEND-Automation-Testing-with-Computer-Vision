from flask import Flask
from jwtauth.database import init_db, shutdown_db_session
import os

SCENES_FOLDER = 'scenes'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['SCENE_FOLDER'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), SCENES_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.teardown_appcontext
def shutdown_session(exception=None):
    shutdown_db_session()


from . import routes

if __name__ == '__main__':
    app.config['SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY')
    init_db()
    app.run()
