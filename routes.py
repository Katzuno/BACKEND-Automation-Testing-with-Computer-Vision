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
    post_body = None
    video = None

    if 'Content-type' in request.headers:
        if request.headers.get('Content-type') == 'application/json':
            post_body = request.get_json(force=True)
        else:
            post_body = request.form
            if "video" not in request.files:
                return jsonify(status=400, message='Video not found in multipart/form-data request')
            video = request.files.get('video')
    else:
        return jsonify(status=403, message='Content-type header not found')

    print(post_body['element_type'])
    scene_id = post_body['scene_id']
    user_id = post_body['user_id']
    element_type = post_body['element_type']

    UPLOAD_PATH = os.path.join('scenes', 'scene_' + str(scene_id) + '_user_' + str(user_id))
    UPLOAD_PATH = os.path.join(UPLOAD_PATH, 'Assets')

    if gui_type != 'cursor' and gui_type != 'video':
        UPLOAD_PATH = os.path.join(UPLOAD_PATH, gui_type)

    if not video:
        if "base64string" in post_body:
            img_data = base64.b64decode(post_body['base64string'])
            filename = element_type + '.jpg'
            with open(os.path.join(UPLOAD_PATH, filename), 'wb') as f:
                f.write(img_data)
        else:
            return jsonify(Error="Image or video is missing")
    else:
        filename = element_type + '-2.mov'
        with open(os.path.join(UPLOAD_PATH, filename), 'wb') as f:
            f.write(video.read())

    return jsonify(success=True, uploaded=gui_type, element_type=element_type)
