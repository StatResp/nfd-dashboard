import os
from config import Config
from flask import Flask
from flask_assets import Environment
from flask import Flask, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin,
                         current_user, login_user, logout_user)
from flask_dance.consumer import oauth_authorized
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.consumer.storage.sqla import (OAuthConsumerMixin,
                                               SQLAlchemyStorage)

# Globally accessible libraries
#TODO - check if this needs to also be initialized with the Dash app?
db = SQLAlchemy()

def init_app():
    """Initialize the core application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'YOUR-SECRET-KEY-HERE'


    # Initialize Plugins - register with flask
    db.init_app(app)




    with app.app_context():
        # import base parts (e.g. python files or logic excluding blueprints)
        from . import routes
        from .statresp.datajoin.cleaning.categorizer import (categorize_numerical_features, Filter_Combo_Builder, FILTER_calculator)
        # import dash app
        from .tdot_dashboard_historical import init_dashboard, protect_views
        app = init_dashboard(app)

        login_manager = LoginManager()
        login_manager.init_app(app)
        login_manager.login_view = 'google.login'

# TODO - factor out sql models ##################################
        class User(db.Model, UserMixin):
            id = db.Column(db.Integer, primary_key=True)
            email = db.Column(db.String(256), unique=True)
            name = db.Column(db.String(256))

        class OAuth(OAuthConsumerMixin, db.Model):
            provider_user_id = db.Column(db.String(256), unique=True)
            user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
            user = db.relationship(User)

        @app.before_first_request
        def create_tables():
            db.create_all()

        @login_manager.user_loader
        def load_user(user_id):
            return User.query.get(int(user_id))
# TODO - factor out sql models ##################################

        # Create & Register Blueprints
        google_blueprint = make_google_blueprint(
            client_id=Config.GOOGLE_OAUTH_CLIENT_ID,
            client_secret=Config.GOOGLE_OAUTH_CLIENT_SECRET,
            scope=['https://www.googleapis.com/auth/userinfo.email',
                   'https://www.googleapis.com/auth/userinfo.profile'],
            offline=True,
            reprompt_consent=True,
            storage=SQLAlchemyStorage(OAuth, db.session, user=current_user)
        )
        app.register_blueprint(google_blueprint)


        # Create or Login local user on successful OAuth Login
        @oauth_authorized.connect_via(google_blueprint)
        def google_logged_in(blueprint, token):
            print(f"[google_logged_in] token: {token}")
            resp = blueprint.session.get('/oauth2/v2/userinfo')
            user_info = resp.json()
            user_id = str(user_info['id'])
            oauth = OAuth.query.filter_by(provider=blueprint.name,
                                          provider_user_id=user_id).first()
            if not oauth:
                # return False
                print(f"[google_logged_in] YOU ARE NOT AUTHORIZED")
                oauth = OAuth(provider=blueprint.name,
                              provider_user_id=user_id,
                              token=token)
            else:
                print(f"[google_logged_in] YES oauth")
                oauth.token = token
                db.session.add(oauth)
                db.session.commit()
                login_user(oauth.user)
            if not oauth.user:
                print(f"[google_logged_in] YOU ARE NOT AUTHORIZED USER")
                user = User(email=user_info["email"],
                            name=user_info["name"])
                oauth.user = user
                db.session.add_all([user, oauth])
                db.session.commit()
                login_user(user)
            return False


        return app