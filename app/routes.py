import os
import pathlib
from flask import current_app as app
from flask import render_template, redirect, url_for, abort, session, request, flash
import requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests
from flask_dance.consumer import oauth_authorized
from flask_login import (LoginManager, UserMixin,
                         current_user, login_user, logout_user)
from flask_dance.contrib.google import make_google_blueprint, google


def login_is_required(function):
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not google.authorized:
            return abort(401)  # Authorization required
        else:
            return function()
    return wrapper


@app.route('/')
def index():
    google_data = None
    user_info_endpoint = 'oauth2/v2/userinfo'
    if current_user.is_authenticated and google.authorized:
        google_data = google.get(user_info_endpoint).json()
        return render_template(
            "index.jinja2",
            title="ScopeLab Dashboards",
            template="home-template",
            body="This is a homepage served with Flask.",
            google_data=google_data,
        )
    else:
        flash('You are not authorized')
        return render_template('index.jinja2',
                           google_data=google_data,
                           fetch_url="https://www.googleapis.com/" + user_info_endpoint)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashapp')
@login_is_required
def dash_render():
    return redirect(url_for('/dashapp/'))



if __name__ == "__main__":
    app.run(debug=True)
