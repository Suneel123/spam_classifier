from collections import OrderedDict
from flask import current_app
from flask import render_template, request, flash, redirect, Blueprint, url_for
from sklearn.model_selection import train_test_split
from spamfilter import spamclassifier
from spamfilter.forms import InputForm
from spamfilter.models import db, File
from werkzeug import secure_filename
import json
import os, re
import pandas as pd
import pickle


spam_api = Blueprint('SpamAPI', __name__)


def allowed_file(filename, extensions=None):
    '''
    'extensions' is either None or a list of file extensions.
    '''
    if filename != '':
        if extensions and isinstance(extensions, list):
            return '.' in filename and filename.rsplit('.', 1)[1] in extensions
        else:
            return '.' in filename and filename.rsplit('.', 1)[1] in ('csv', 'CSV')
    return False


@spam_api.route('/')
def index():
    '''
    Renders 'index.html'
    '''
    return render_template('index.html')


@spam_api.route('/listfiles/<success_file>/')
@spam_api.route('/listfiles/')
def display_files(success_file=None):
    '''
    Obtain the filenames of all CSV files present in 'inputdata' folder and
    pass it to template variable 'files'.

    if 'success_file' value is passed, corresponding file is highlighted.
    '''
    files = [file for file in os.listdir(current_app.config['INPUT_DATA_UPLOAD_FOLDER'])
                   if file.rsplit('.', 1)[-1].lower() == 'csv']
    if success_file:
        return render_template('fileslist.html', files=files, fname=success_file)
    else:
        return render_template('fileslist.html', files=files)


def validate_input_dataset(input_dataset_path):
    '''
    Validate the following details of an Uploaded CSV file

    1. The CSV file must contain only 2 columns. If not display the below error message.
    'Only 2 columns allowed: Your input csv file has '+<No_of_Columns_found>+ ' number of columns.'

    2. The column names must be "text" nad "spam" only. If not display the below error message.
    'Differnt Column Names: Only column names "text" and "spam" are allowed.'

    3. The 'spam' column must conatin only integers. If not display the below error message.
    'Values of spam column are not of integer type.'

    4. The values of 'spam' must be either 0 or 1. If not display the below error message.
    'Only 1 and 0 values are allowed in spam column: Unwanted values ' + <Unwanted values joined by comma> + ' appear in spam column'

    5. The 'text' column must contain string values. If not display the below error message.
    'Values of text column are not of string type.'

    6. Every input email must start with 'Subject:' pattern. If not display the below error message.
    'Some of the input emails does not start with keyword "Subject:".'

    Return False if any of the above 6 validations fail.

    Return True if all 6 validations pass.
    '''
    input_df = pd.read_csv(input_dataset_path)
    columns = list(input_df.columns)
    if len(columns) != 2:
        return False, "Only 2 columns allowed: Your input csv file has '{}' number of columns.".format(columns)
    if not "text" in columns or not "spam" in columns:
        return False, 'Differnt Column Names: Only column names "text" and "spam" are allowed.'
    if input_df.spam.dtype != 'int64':
        return False, 'Values of spam column are not of integer type.'
    not_spam_labels = list(input_df[~input_df.spam.isin([0, 1])].values[:, 1])
    if len(not_spam_labels):
        return False, "Only 1 and 0 values are allowed in spam column: Unwanted values '{}' appear in " \
                "spam column".format(','.join(not_spam_labels))
    if not input_df.applymap(type).eq(str).all()['text']:
        return False, 'Values of text column are not of string type.'
    expn = re.compile(r"^Subject:.*")
    if len(input_df[~input_df.text.apply(lambda x: bool(expn.match(x)))].index):
        return False, 'Some of the input emails does not start with keyword "Subject:".'
    return True, None


@spam_api.route('/upload/', methods=['GET', 'POST'])
def file_upload():
    '''
    If request is GET, Render 'upload.html'

    If request is POST, capture the uploaded file and check if the uploaded file is 'csv' extension, using 'allowed_file' defined above.

    if 'allowed_file' returns False, display the below error message and redirect to 'upload.html' with GET request.
    'Only CSV Files are allowed as Input.'

    if 'allowed_file' returns True, save the file in 'inputdata' folder and
    validate the uploaded csv file using 'validate_input_dataset' defined above.

    if 'validate_input_dataset' returns 'False', remove the file from 'inputdata' folder,
    redirect to 'upload.html' with GET request and respective error message.

    if 'validate_input_dataset' returns 'True', create a 'File' object and save it in database, and
    render 'display_files' template with template varaible 'success_file', set to filename of uploaded file.

    '''
    if request.method == 'POST':
        if 'uploadfile' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['uploadfile']
        if not allowed_file(file.filename):
            flash('Only CSV Files are allowed as Input.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], filename)
            file.save(file_path)
            valid_dataset, msg = validate_input_dataset(file_path)
            if not valid_dataset:
                os.remove(file_path)
                flash(msg)
                return redirect(request.url)
            else:
                file_obj = File(name=filename, filepath=file_path)
                db.session.add(file_obj)
                db.session.commit()
                return redirect(url_for('SpamAPI.display_files', success_file=filename))
    return render_template('upload.html')


def validate_input_text(intext):
    '''
    Validate the following details of input email text, provided for prediction.

    1. If the input email text contains more than one mail, they must be separated by atleast one blank line.

    2. Every input email must start with 'Subject:' pattern.

    Return False if any of the two validations fail.

    If all valiadtions pass, Return an Ordered Dicitionary, whose keys are first 30 characters of each
    input email and values being the complete email text.
    '''
    input_emails = intext.split("\n\n")
    expn = re.compile(r"^Subject:.*")
    bool_list = [bool(expn.match(email.strip())) for email in input_emails]
    if not all(bool_list):
        return False
    else:
        ord_dict = OrderedDict()
        for email in input_emails:
            ord_dict[email.strip()[:30]] = email.strip()
        return ord_dict


@spam_api.route('/models/<success_model>/')
@spam_api.route('/models/')
def display_models(success_model=None):
    '''
    Obtain the filenames of all machine learning models present in 'mlmodels' folder and
    pass it to template variable 'files'.

    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.

    Consider only the model and not the word_features.pk files.

    Renders 'modelslist.html' template with values  of varaibles 'files' and 'model_name'.
    'model_name' is set to value of 'success_model' argument.
    if 'success_model value is passed, corresponding model file name is highlighted.
    '''
    files = os.listdir(current_app.config['ML_MODEL_UPLOAD_FOLDER'])
    models = [model_file for model_file in files if '_'.join(model_file.rsplit("_", 2)[-2:])!="word_features.pk"]
    return render_template('modelslist.html', files=models, model_name=success_model)


def isFloat(value):
    '''
    Return True if <value> is a float, else return False
    '''
    try:
        float(value)
        if value.isdigit():
            return False
        else:
            return True
    except ValueError:
        return False


def isInt(value):
    '''
    Return True if <value> is an integer, else return False
    '''
    try:
        float(value)
        if value.isdigit():
            return True
        else:
            return False
    except ValueError:
        return False


@spam_api.route('/train/', methods=['GET', 'POST'])
def train_dataset():
    '''
    If request is of GET method, render 'train.html' template with tempalte variable 'train_files',
    set to list if csv files present in 'inputdata' folder.

    If request is of POST method, capture values associated with
    'train_file', 'train_size', 'random_state', and 'shuffle'

    if no 'train_file' is selected, render the same page with GET Request and below error message.
    'No CSV file is selected'

    if 'train_size' is None, render the same page with GET Request and below error message.
    'No value provided for size of training data set.'

    if 'train_size' value is not float, render the same page with GET Request and below error message.
    'Training Data Set Size must be a float.

    if 'train_size' value is not in between 0.0 and 1.0, render the same page with GET Request and below error message.
    'Training Data Set Size Value must be in between 0.0 and 1.0'

    if 'random_state' is None,render the same page with GET Request and below error message.
    'No value provided for random state.''

    if 'random_state' value is not an integer, render the same page with GET Request and below error message.
    'Random State must be an integer.'

    if 'shuffle' is None, render the same page with GET Request and below error message.
    'No option for shuffle is selected.'

    if 'shuffle' is set to 'No' when 'Startify' is set to 'Yes', render the same page with GET Request and below error message.
    'When Shuffle is No, Startify cannot be Yes.'

    If all input values are valid, build the model using submitted paramters and methods defined in
    'spamclassifier.py' and save the model and model word features file in 'mlmodels' folder.

    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.

    Finally render, 'display_models' template with value of template varaible 'success_model'
    set to name of model generated, ie. 'sample.pk'
    '''
    train_files = [file for file in os.listdir(current_app.config['INPUT_DATA_UPLOAD_FOLDER'])
                   if file.rsplit('.', 1)[-1].lower() == 'csv']
    if request.method == 'POST':
        if 'train_file' not in request.form:
            flash('No CSV file is selected')
            return redirect(request.url)
        train_file = request.form['train_file']
        train_size = request.form.get('train_size')
        if train_size is None:
            flash('No value provided for size of training data set.')
            return redirect(request.url)
        if not isFloat(train_size):
            flash('Training Data Set Size must be a float.')
            return redirect(request.url)
        train_size = float(train_size)
        if not (0.0 <= train_size <= 1.0):
            flash('Training Data Set Size Value must be in between 0.0 and 1.0')
            return redirect(request.url)
        random_state = request.form.get('random_state')
        if random_state is None:
            flash('No value provided for random state.')
            return redirect(request.url)
        if not isInt(random_state):
            flash('Random State must be an integer.')
            return redirect(request.url)
        if 'shuffle' not in request.form:
            flash('No option for shuffle is selected.')
            return redirect(request.url)
        shuffle = request.form['shuffle']
        stratify = request.form['stratify']
        if shuffle=='N' and stratify == 'Y':
            flash('When Shuffle is No, Startify cannot be Yes.')
            return redirect(request.url)
        shuffle = shuffle == 'Y'
        stratify = stratify == 'Y'
        df = pd.read_csv(os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], train_file))
        if not shuffle:
            train_X, test_X, train_Y, test_Y = train_test_split(df["text"].values, df["spam"].values,
                                                                train_size=train_size, shuffle=shuffle)
        if shuffle:
            if stratify:
                train_X, test_X, train_Y, test_Y = train_test_split(df["text"].values, df["spam"].values,
                                                                    train_size=train_size, shuffle=shuffle,
                                                                    stratify=df["spam"].values)

            if not stratify:
                train_X, test_X, train_Y, test_Y = train_test_split(df["text"].values, df["spam"].values,
                                                                    train_size=train_size, shuffle=shuffle)
        classifier = spamclassifier.SpamClassifier()
        classifier_model, classifier_word_features = classifier.train(train_X, train_Y)
        train_file_name = train_file.rsplit(".",1)[0]
        model_name = train_file_name + '.pk'
        model_word_features_name = train_file_name + '_word_features.pk'
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_name)
        model_wf_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_word_features_name)
        with open(model_file, 'wb') as model_fp:
            pickle.dump(classifier_model, model_fp)
        with open(model_wf_file, 'wb') as model_wfp:
            pickle.dump(classifier_word_features, model_wfp)
        return redirect(url_for('SpamAPI.display_models', success_model=model_name))
    return render_template('train.html', train_files=train_files)


@spam_api.route('/results/')
def display_results():
    '''
    Read the contents of 'predictions.json' and pass those values to 'predictions' template varaible

    Render 'displayresults.html' with value of 'predictions' template variable.
    '''
    predictions_fpath = os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], 'predictions.json')
    with open(predictions_fpath, 'r') as fp:
       predictions = json.load(fp, object_pairs_hook=OrderedDict)
    return render_template('displayresults.html', predictions=list(predictions.items()))


@spam_api.route('/predict/', methods=['GET', "POST"])
def predict():
    '''
    If request is of GET method, render 'emailsubmit.html' template with value of template
    variable 'form' set to instance of 'InputForm'(defined in 'forms.py').
    Set the 'inputmodel' choices to names of models (in 'mlmodels' folder), with out extension i.e .pk

    If request is of POST method, perform the below checks

    1. If input emails is not provided either in text area or as a '.txt' file, render the same page with GET Request and below error message.
    'No Input: Provide a Single or Multiple Emails as Input.'

    2. If input is provided both in text area and as a file, render the same page with GET Request and below error message.
    'Two Inputs Provided: Provide Only One Input.'

    3. In case if input is provided as a '.txt' file, save the uploaded file into 'inputdata' folder and read the
     contents of file into a variable 'input_txt'

    4. If input provided in text area, capture the contents in the same variable 'input_txt'.

    5. validate 'input_txt', using 'validate_input_text' function defined above.

    6. If 'validate_input_text' returns False, render the same page with GET Request and below error message.
    'Unexpected Format : Input Text is not in Specified Format.'

    7. If 'validate_input_text' returns a Ordered dictionary, choose a model and perform prediction of each input email using 'predict' method defined in 'spamclassifier.py'

    8. If no input model is choosen, render the same page with GET Request and below error message.
    'Please Choose a single Model'

    9. Convert the ordered dictionary of predictions, with 0 and 1 values, to another ordered dictionary with values 'NOT SPAM' and 'SPAM' respectively.

    10. Save thus obtained predictions ordered dictionary into 'predictions.json' file.

    11. Render the template 'display_results'
    '''
    form = InputForm()
    files = os.listdir(current_app.config['ML_MODEL_UPLOAD_FOLDER'])
    form.inputmodel.choices = [(model_file.rsplit('.',1)[0], model_file.rsplit('.',1)[0])
                               for model_file in files
                               if '_'.join(model_file.rsplit("_", 2)[-2:])!="word_features.pk"]
    if request.method == 'POST':
        inputemail = request.form.get('inputemail')
        if len(request.files) > 0:
            inputfile = request.files[form.inputfile.name]
        else:
            inputfile = None
        inputmodel = request.form.get('inputmodel')
        if (inputemail == '' or inputemail is None) and inputfile is None:
            flash('No Input: Provide a Single or Multiple Emails as Input.')
            return redirect(url_for('SpamAPI.predict'))
        if not (inputemail == '' or inputemail is None) and not inputfile is None:
            flash('Two Inputs Provided: Provide Only One Input.')
            return redirect(url_for('SpamAPI.predict'))
        if inputemail:
            input_txt = inputemail
        if inputfile is not None:
            inputfile_path = os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], 'input.txt')
            inputfile.save(inputfile_path)
            with open(inputfile_path, 'r') as fp:
                input_txt = fp.read()
        emails_dict = validate_input_text(input_txt)
        if not emails_dict:
            flash('Unexpected Format : Input Text is not in Specified Format.')
            return redirect(url_for('SpamAPI.predict'))
        if inputmodel is None:
            flash('Please Choose a single Model')
            return redirect(url_for('SpamAPI.predict'))
        sc = spamclassifier.SpamClassifier()
        sc.load_model(inputmodel.rsplit('.', 1)[0])
        emails_pred = sc.predict(emails_dict)
        emails_pred_final = OrderedDict()
        for email, pred in emails_pred.items():
            if pred:
                emails_pred_final[email] = 'SPAM'
            else:
                emails_pred_final[email] = 'NOT SPAM'
        predictions_fpath = os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], 'predictions.json')
        with open(predictions_fpath, 'w') as fp:
            json.dump(emails_pred_final, fp)
        return redirect(url_for('SpamAPI.display_results'))
    return render_template('emailsubmit.html', form=form)
