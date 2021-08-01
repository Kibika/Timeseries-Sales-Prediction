"""Routes for parent Flask app."""
from flask import current_app as app
from flask import render_template


@app.route("/upload_csv", methods=["GET", "POST"])
def upload_csv():
    return render_template("upload_csv.html")

@app.route("/sales_prediction")
def sales_prediction():
    return render_template("public/default.html",content="prediction_layout.html", title='Dashboard | Sales Prediction')

@app.route("/")
def home_view():
        return render_template("upload_csv.html")