from os import environ, path

from dotenv import load_dotenv

BASE_DIR = path.abspath(path.dirname(__file__))
load_dotenv(path.join(BASE_DIR, ".env"))

class Config:
    FLASK_APP = environ.get("FLASK_APP")
    FLASK_RUN_PORT = environ.get("FLASK_RUN_PORT")
    FLASK_ENV = environ.get("FLASK_ENV")
    SECRET_KEY = environ.get("SECRET_KEY")
    environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
    GOOGLE_OAUTH_CLIENT_ID = environ.get("GOOGLE_OAUTH_CLIENT_ID")
    GOOGLE_OAUTH_CLIENT_SECRET = environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
    MAPBOX_ACCESS_TOKEN="pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"
    LATINITIAL = 36.16228
    LONINITAL = -86.774372