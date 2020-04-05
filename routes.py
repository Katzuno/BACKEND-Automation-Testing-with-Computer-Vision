import os

from flask_jwt import jwt_required
from jwtauth.jwt import *
from flask_jwt import JWT
import json
from flask import flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from . import app, allowed_file
import base64


@app.route('/hello')
def hello():
    return 'Hello world!'


@app.route('/auth', methods=['POST'])
def auth():
    return 'Hello world'


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route('/upload/<gui_type>', methods=['POST'])
def api_root(gui_type):
    post_body = request.get_json(force=True)
    if "base64string" in post_body:
        scene_id = post_body['scene_id']
        user_id = post_body['user_id']
        element_type = post_body['element_type']

        UPLOAD_PATH = os.path.join('scenes', 'scene_' + str(scene_id) + '_user_' + str(user_id))
        UPLOAD_PATH = os.path.join(UPLOAD_PATH, 'Assets', gui_type)

        imgdata = base64.b64decode(post_body['base64string'])
        filename = element_type + '.jpg'
        with open(os.path.join(UPLOAD_PATH, filename), 'wb') as f:
            f.write(imgdata)

        return json.dumps({'success': True})
    else:
        return json.dumps({"Error": "Img_url is missing"})



    """
    app.logger.info(UPLOAD_PATH)
    img = request.files['image']
    img_name = secure_filename(img.filename)
    create_new_folder(UPLOAD_PATH)
    saved_path = os.path.join(UPLOAD_PATH, img_name)
    app.logger.info("saving {}".format(saved_path))
    img.save(saved_path)
    return jsonify({'success': True})
    """
