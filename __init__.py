from flask import Flask
from flaskext.mysql import MySQL
from flask_cors import CORS
import sqlite3
from jwtauth.database import init_db, shutdown_db_session
import os

SCENES_FOLDER = 'scenes'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
CORS(app)
app.config['SCENE_FOLDER'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), SCENES_FOLDER)

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'licenta'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_THREAD_SAFETY'] = 1
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()


def insert(fields_to_insert="", table_name="", value1="", value2=""):
    str_to_format = "INSERT INTO %s(%s) VALUES ('%d','%s')"
    sql_query = str_to_format % (table_name, fields_to_insert, value1, value2)
    print(sql_query)
    if cursor.execute(sql_query):
        conn.commit()
        return {'Status': 'Success'}
    else:
        return ""


def select(fields_selected="", table_name="", where_clause="", limit=""):
    sql_query = "SELECT " + fields_selected + " FROM " + table_name + " WHERE " + where_clause + " LIMIT " + limit
    if limit == "":
        sql_query = sql_query[:-7]
    if where_clause == "":
        sql_query = sql_query[:-7]
    cursor.execute(sql_query)
    return cursor.fetchall()


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
