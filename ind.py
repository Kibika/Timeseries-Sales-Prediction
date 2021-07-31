"""Application entry point."""
from whitenoise import WhiteNoise
from apps import init_app

app = init_app()

if __name__ == "__main__":
    app.run(debug=False)
    app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
    # app.run(host="0.0.0.0")
