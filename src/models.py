from os import name

from . import db 
from flask_login import UserMixin
from datetime import datetime
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    last_name = db.Column(db.String(150))
    images = db.relationship('Image', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.first_name}', '{self.last_name}', '{self.email}')"
    
    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    relational_id = db.Column(db.Integer)
    data = db.Column(db.String(100))
    date = db.Column(db.DateTime(timezone=True), default=datetime.now())
    title = db.Column(db.String(150))
    description = db.Column(db.String(500))
    labels = db.Column(db.String(200))
    quantity = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __repr__(self):
        return f"Image('{self.data}', '{self.date}')"

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
    