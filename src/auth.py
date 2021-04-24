from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user

auth = Blueprint("auth", __name__)


@auth.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user:
            if check_password_hash(user.password, password):
                flash("Logged in", category="success")
                login_user(user, remember=True)
                return redirect(url_for("views.home"))
            else:
                flash("Incorrect password", category="error")
        else:
            flash("Incorrect email", category="error")
    return render_template("login.html", user=current_user)


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


@auth.route("/sign-up", methods=['GET','POST'])
def signUp():
    if request.method == "POST":
        firstName = request.form.get("FirstName")
        lastName = request.form.get("LastName")
        email = request.form.get("Email")
        password = request.form.get("Password")
        passwordConf = request.form.get("PasswordConf")

        user = User.query.filter_by(email=email).first()
        if user:
            flash("This email already exists!", category="error")
        elif len(firstName) < 2:
            flash("First Name must be at least 2 characters", category="error")
        elif len(lastName) < 2:
            flash("Last Name must be at least 2 characters", category="error")
        elif len(email) < 4:
            flash("Email must be at least 4 characters", category="error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters", category="error")
        elif password != passwordConf:
            flash("Password is not the same", category="error")
        else:
            new_user = User(first_name=firstName, last_name=lastName, email=email,
                            password=generate_password_hash(password, method="sha256"))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash("Welcome in!", category="success")
            return redirect(url_for('views.home'))

    return render_template("signUp.html",user=current_user)
