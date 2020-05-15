import os

from flask_jwt import jwt_required
from jwtauth.jwt import *
from flask_jwt import JWT
import json
from flask import flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from . import app, allowed_file, insert, select
import base64
import subprocess


@app.route('/auth', methods=['POST'])
def auth():
    return 'Hello world'


@app.route('/create/scene', methods=['POST'])
def create_scene():
    post_body = None

    if 'Content-type' in request.headers:
        if request.headers.get('Content-type') == 'application/json':
            post_body = request.get_json(force=True)
    else:
        return jsonify(status=403, message='Content-type header not found')

    sceneName = post_body['scene_name']
    userId = int(post_body['user_id'])
    if insert(fields_to_insert="user_id, scene_name", table_name='SCENES', value1=userId, value2=sceneName):
        return jsonify(status=201, message='Created')
    return jsonify(status=505, message='SQL error')


@app.route('/get/scenes', methods=['POST'])
def get_scenes():
    post_body = None

    if 'Content-type' in request.headers:
        if request.headers.get('Content-type') == 'application/json':
            post_body = request.get_json(force=True)
    else:
        return jsonify(status=403, message='Content-type header not found')

    userId = int(post_body['user_id'])
    scenesList = select(fields_selected="scene_name", table_name='SCENES')
    scenes = []
    for scene in scenesList:
        scenes.append(scene[0])

    if scenesList:
        return jsonify(status=201, scenes=scenes)

    return jsonify(status=505, message='SQL error')


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route('/run/scene', methods=['POST'])
def run_scene():
    post_body = None
    if 'Content-type' in request.headers:
        if request.headers.get('Content-type') == 'application/json':
            post_body = request.get_json(force=True)
    else:
        return jsonify(status=403, message='Content-type header not found')

    userId = post_body['user_id']
    sceneId = post_body['scene_id']
    sceneFolder = os.path.join(app.config['SCENE_FOLDER'], 'scene_' + str(sceneId) + '_user_' + str(userId))
    assetsFolder = os.path.join(sceneFolder, 'Assets')
    process = subprocess.Popen(['python', app.config['ATCV_FILE'], assetsFolder],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    return jsonify(status=201, message='Scene finished, please refresh!')


@app.route('/upload/<gui_type>', methods=['POST'])
def upload_gui(gui_type):
    post_body = None
    video = None

    if 'Content-type' in request.headers:
        if request.headers.get('Content-type') == 'application/json':
            post_body = request.get_json(force=True)
        else:
            post_body = request.form
            if "video" not in request.files:
                return jsonify(status=409, message='Video not found in multipart/form-data request')
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
            filename = element_type + '.png'
            create_new_folder(UPLOAD_PATH)
            with open(os.path.join(UPLOAD_PATH, filename), 'wb') as f:
                f.write(img_data)
        else:
            return jsonify(Error="Image or video is missing")
    else:
        filename = element_type + '.mov'
        create_new_folder(UPLOAD_PATH)
        with open(os.path.join(UPLOAD_PATH, filename), 'wb') as f:
            f.write(video.read())

    return jsonify(success=True, uploaded=gui_type, element_type=element_type)


@app.route('/upload/actions/<input_type>', methods=['POST'])
def upload_functions(input_type):
    post_body = None
    video = None

    if 'Content-type' in request.headers:
        if request.headers.get('Content-type') == 'application/json':
            post_body = request.get_json(force=True)
            input = post_body['actions']

            scene_id = post_body['scene_id']
            user_id = post_body['user_id']

            UPLOAD_PATH = os.path.join('scenes', 'scene_' + str(scene_id) + '_user_' + str(user_id))
            UPLOAD_PATH = os.path.join(UPLOAD_PATH, 'Assets')

            filename = input_type + '.txt'

            with open(os.path.join(UPLOAD_PATH, filename), 'w+') as f:
                for row in input:
                    f.write(row)

    return jsonify(success=True, uploaded=input_type, file_path=os.path.join(UPLOAD_PATH, filename))
