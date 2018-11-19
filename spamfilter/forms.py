from wtforms import Form, TextAreaField, SubmitField, FileField, validators, RadioField
# from flask_wtf import FlaskForm


class InputForm(Form):

    inputemail = TextAreaField("Input Email")
    inputfile = FileField("Input File")
    inputmodel = RadioField("Choose a Model:")
    submit = SubmitField("Submit")
