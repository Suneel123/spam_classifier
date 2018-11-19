import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(BASE_DIR,'app.db')
SQLALCHEMY_TRACK_MODIFICATIONS=False
SECRET_KEY='prod'
ML_MODEL_UPLOAD_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'spamfilter/mlmodels')
INPUT_DATA_UPLOAD_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'spamfilter/inputdata')
ALLOWED_EXTENSIONS = set(['pk', 'txt', 'csv'])
MAX_CONTENT_PATH = 16*1024*1024
