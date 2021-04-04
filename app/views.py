# !/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import render_template
import subprocess
from app import app

SCRIPTS_OUTPUT = {'training': '',
                  'testing': ''}


@app.route('/')
def hello_world():
    SCRIPTS_OUTPUT['training'] = ''
    SCRIPTS_OUTPUT['testing'] = ''
    return render_template("index.html",
                           scripts_output=SCRIPTS_OUTPUT
                           )


@app.route('/train')
def train():
    ps = subprocess.run(["python", "./scripts/train.py"], capture_output=True)
    SCRIPTS_OUTPUT['training'] = str(ps.stdout)
    # return ps.stdout
    return render_template("index.html",
                           scripts_output=SCRIPTS_OUTPUT
                           )


@app.route('/test')
def test():
    ps = subprocess.run(["python", "./scripts/test.py"], capture_output=True)
    SCRIPTS_OUTPUT['testing'] = str(ps.stdout)
    # return ps.stdout
    return render_template("index.html",
                           scripts_output=SCRIPTS_OUTPUT
                           )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
