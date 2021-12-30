# Incident-dashboard

If you do not have this environment then create it. To run just follow the instructions based on docker as described  below.

## Running app with Docker

```
docker build -t tdotdash .
docker run --memory='4g' --rm -p 8080:8080 tdotdash
```

Note we give 4 gigabyte to this. Make sure your docker dashboard has enough resources enabled. 

Go to your browser and open http://127.0.0.1:8080. It should look like below.

<img src="docs/dashboard.png"
     alt="Dashboard image"
     style="float: left; margin-right: 0px;" />

# Authentication Changes/Notes
- Flask server running
- Dash piggybacking as a module
- Pass Flask app server into create_dashboard()
- Allows Dash to use own Flask instance instead of its own

Todo:
- Test with Docker (only tested locally as of 12.29)
- Create roles for users
-    Note: flask user may be a different paradigm than currently used flask-login/dance but premise is similar
-    https://flask-user.readthedocs.io/en/latest/authorization.html
-    https://flask-user.readthedocs.io/en/latest/basic_app.html
- Allow admin role to add users to whitelist
- Only allow whitelisted google authorized users access to dashboard
- Incorporate other authorizations (e.g. OICD, ...) 

Issues
- When testing locally had to run pip install -r requirements.txt twice [error: invalid command 'bdist_wheel']
- This related to easydev, colormap, and dash-daq (12.29)
