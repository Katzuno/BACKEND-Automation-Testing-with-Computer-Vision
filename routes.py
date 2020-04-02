from flask_jwt import jwt_required
from jwtauth.jwt import *
from flask_jwt import JWT


from . import app

@app.route('/hello')
def hello():
    return 'Hello world!'

@app.route('/auth', methods = ['POST'])
def auth():
    return 'Hello world'
