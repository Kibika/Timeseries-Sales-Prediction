"""Application entry point."""
from apps import init_app

app = init_app()

if __name__ == "__main__":
    app.run(debug=False)
    # app.run(host="0.0.0.0")