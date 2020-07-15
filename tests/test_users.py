# ~/movie-bag/tests/test_signup.py
import os
import sys

topdir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(topdir)

import unittest
import json

import requests

DOMAIN_NAME = 'localhost'
PORT = '5000'


class UserTest(unittest.TestCase):
    base_uri = 'http://' + DOMAIN_NAME + ':' + PORT + '/'

    def test_get_scene(self):
        # Given
        payload = json.dumps({
            "user_id": "2"
        })
        # When
        response = requests.post(self.base_uri + 'get/scenes', headers={"Content-Type": "application/json"}, data=payload)

        # Then
        print(response)
        self.assertEqual(201, response.json()['status'])
        self.assertEqual(dict, type(response.json()))
        self.assertEqual(200, response.status_code)

    def test_get_scene_output(self):
        # Given
        payload = json.dumps({
            "user_id": 2,
            "scene_id": 3
        })
        # When
        response = requests.post(self.base_uri + 'get/scene/output', headers={"Content-Type": "application/json"}, data=payload)

        # Then
        print(response)
        self.assertEqual(str, type(response.json()['output']))
        self.assertEqual(200, response.status_code)
