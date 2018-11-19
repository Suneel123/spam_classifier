from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

class File(db.Model):
    '''
    id - Which stores a file id
    name - Which stores the file name
    filepath - Which stores path of file,
    '''
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300), index=True)
    filepath = db.Column(db.String(500), index=True, unique=True)

    def __repr__(self):
        return "<File : {}>".format(self.name)
