from flask import Flask
from jwtauth.database import init_db, shutdown_db_session
from os import environ

app = Flask(__name__)


@app.teardown_appcontext
def shutdown_session(exception=None):
    shutdown_db_session()


from . import routes

if __name__ == '__main__':
    app.config['SECRET_KEY'] = environ.get('JWT_SECRET_KEY')
    init_db()
    app.run()
