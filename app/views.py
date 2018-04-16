import os
import io
from datetime import datetime, timedelta
import uuid

from flask import flash, render_template, request, redirect, url_for, jsonify, send_from_directory, abort
from flask_admin.contrib.sqla import ModelView
from flask_admin.form.upload import FileUploadField
from flask_admin import AdminIndexView
import flask_security as security
from flask_security.utils import encrypt_password
import flask_login as login
from flask_login import login_required
from wtforms.fields import PasswordField
from sqlalchemy import Date, cast

import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import base64

from app import app, security, db
from models import User, Competition, Submission


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submission', methods=['GET', 'POST'])
@login_required
def upload_file():

    try:

        now = datetime.now()

        competitions = [c for c in Competition.query.all() if (not c.start_on or
            c.start_on <= now) and (not c.end_on or c.end_on >= now)]

        if request.method == 'POST':

            competition_id = request.form.get('competitions')
            if competition_id is None:
                flash('No competition selected')
                return redirect(request.url)

            user_id = login.current_user.id

            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            # check if the user has made submissions in the past 24h
            if Submission.query.filter_by(user_id=user_id).filter_by(competition_id=competition_id).filter(Submission.submitted_on>now-timedelta(hours=1)).count() > 0:
                flash("Tu as déjà fait une soumission il y a moins d'une heure " + user_id)
                return redirect(request.url)

            if file:

                filename = str(uuid.uuid4()) + ".csv"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # save submission
                submission = Submission()
                submission.user_id = login.current_user.id
                submission.competition_id = competition_id
                submission.filename = filename
                (submission.preview_score, submission.score) = get_scores(filepath, competition_id)
                submission.submitted_on = now.replace(microsecond=0)
                submission.comment = request.form.get("comment")
                db.session.add(submission)
                db.session.commit()

                return redirect(url_for('scores'))

        return render_template('submission.html', competitions=competitions)

    except ParsingError as e:
        flash(str(e))
        return redirect(request.url)


# @login_required
def get_scores(filename, competition_id):
    "Returns (preview_score, score)"

    regex = r'(\d+),(.+)'

    # parse files
    # filename = "C:\\Users\\jerem\\Documents\\ESTIAM\\UE Datascience\\test_only_labels.csv"
    predictions = np.fromregex(filename, regex, [('id', np.int64),
                                                 ('v0', 'S128')])
    groundtruth_filename = os.path.join(
            app.config['GROUNDTRUTH_FOLDER'],
            Competition.query.get(competition_id).groundtruth)
    groundtruth = np.fromregex(
            groundtruth_filename,
            regex,
            [('id', np.int64), ('v0', 'S128')])

    # sort data
    predictions.sort(order='id')
    groundtruth.sort(order='id')

    if predictions['id'].size == 0 or not np.array_equal(predictions['id'],
                                                         groundtruth['id']):
        raise ParsingError("Error parsing the submission file. Make sure it" +
                           "has the right format and contains the right ids.")

    # partition the data indices into two sets and evaluate separately
    splitpoint = int(np.round(len(groundtruth) * 0.15))
    score_p = accuracy_score(groundtruth['v0'][:splitpoint],
                             predictions['v0'][:splitpoint])
    score_f = accuracy_score(groundtruth['v0'][splitpoint:],
                             predictions['v0'][splitpoint:])

    return (score_p, score_f)


@app.route('/scores', methods=['GET', 'POST'])
@login_required
def scores():
    competitions = Competition.query.all()
    return render_template('scores.html', competitions=competitions)


# @app.route("/plots")
def plot_confusion_matrix(user_id):
    print("USER_ID : " + str(user_id))
    cmap = plt.cm.Blues
    normalize = False
    title = 'Confusion matrix'
    regex = r'(\d+),(.+)'
    competition_id = 0
    classes = ["functional", "non functional", "functional needs repair"]
    competitions = Competition.query.all()
    for c in competitions:
        if c.name == "ESTIAM 2018":
            competition_id = c.id
    # users = User.query.all()
    # username = ""
    score = 0
    filepath = ""
    print("COMPETITION_ID : " + str(competition_id))
    # for u in users:
    #     if u.user_id == user_id:
    #         username = u.username
    submissions = Submission.query.all()
    i = 0
    for s in submissions:
        i += 1
        print(i)
        print("S_USER_id : " + str(s.user_id) + " | COMPETITION_ID : " +
              str(s.competition_id))
        print(str(user_id)+str(s.user_id)+"HA" + " | " + str(competition_id) +
              str(s.competition_id) + "HO")
        if s.user_id == user_id and s.competition_id == competition_id:
            print("SCORE : " + s.score)
            if s.score > score:
                filepath = os.path.join(
                        "/home/ubuntu/submission/app/files/upload/",
                        s.filename)
                score = s.score

    print("SCORE : " + str(score))
    print("FILEPATH : " + filepath)
    # parse files
    # filename = "C:\\Users\\jerem\\Documents\\ESTIAM\\UE Datascience" +
    # "\\test_only_labels.csv"
    predictions = np.fromregex(filepath, regex, [('id', np.int64),
                                                 ('v0', 'S128')])
    groundtruth_filename = os.path.join(
            "/home/ubuntu/submission/app/files/groundtruth/",
            Competition.query.get(competition_id).groundtruth)
    groundtruth = np.fromregex(
            groundtruth_filename,
            regex,
            [('id', np.int64), ('v0', 'S128')])
    cm = confusion_matrix(groundtruth['v0'], predictions['v0'])
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    img = io.BytesIO()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    print("PLOT_URL : " + plot_url)

    return '<img src="data:image/png;base64,{}">'.format(plot_url)


@app.route('/plots', methods=['GET', 'POST'])
@login_required
def plots():
    users = User.query.all()
    return render_template('plots.html', users=users)


@app.route('/_get_datas', methods=['POST'])
@login_required
def get_users():
    if request.method == 'POST':
        user_id = request.form.get('users')
        # submissions = Submission.query.filter(Submission.competition_id==competition_id)
        s = plot_confusion_matrix(user_id)
        return jsonify({"count": 1, "s": s})


@app.route('/_get_submissions', methods=['POST'])
@login_required
def get_submissions():
    if request.method == 'POST':
        competition_id = request.form.get('competitions')
        submissions = Submission.query.filter(Submission.competition_id==competition_id)
#        if not login.current_user.has_role('admin'):
#            submissions = submissions.filter_by(user_id=login.current_user.id)

        count = submissions.count()

        if count == 0:
            return jsonify({"count": 0})

        response = {}

        # get all users
        user_ids = sorted(list({s.user_id for s in submissions}))
        dates = sorted(list({s.submitted_on for s in submissions}))

        rows = ""
        for d in dates:
            row = '{{"c":[{{"v":"Date({0},{1},{2},{3},{4},{5})"}}'.format(d.year, d.month - 1, d.day, d.hour, d.minute, d.second)
            for u in user_ids:
                s = submissions.filter(Submission.submitted_on==d).filter(Submission.user_id==u)
                if s.count() > 0:
                    score = s.first().preview_score * 100
                    row += ',{{"v":{:.2f}}}'.format(score)
                    row += ',{{"v":"<div style=\\"padding:5px\\"><b>Date</b>: {}<br><b>Username</b>: {}<br><b>Score</b>: {:.2f}<br><b>Comment</b>: {}</div>"}}'.format(
                                s.first().submitted_on,
                                User.query.get(u).username,
                                score,
                                s.first().comment)
                else:
                    row += ',{"v":"null"}'
                    row += ',{"v":"null"}'

            row += "]},"
            rows += row


        s = """
        {{
          "cols": [
                {{"id":"","label":"Date","pattern":"","type":"datetime"}},
                {0}
              ],
          "rows": [
                {1}
              ]
        }}""".format(
                ','.join('{{"id":"","label":"{}","pattern":"","type":"number"}},{{"id":"","label":"Comment","pattern":"","type":"string","role":"tooltip","p":{{"html":true}}}}'.format(User.query.get(u).username) for u in user_ids),
                rows
                )

        return jsonify({"count": count, "s": s})


###############
# Admin views #
###############

class MyAdminIndexView(AdminIndexView):
    def is_accessible(self):
        return login.current_user.has_role('admin')


class AdminModelView(ModelView):

    can_set_page_size = True
    can_export = True

    def is_accessible(self):
        return login.current_user.has_role('admin')

#    def inaccessible_callback(self, name, **kwargs):
#        # redirect to login page if user doesn't have access
#        return redirect(url_for('security.login', next=request.url))


class UserAdmin(ModelView):

    # Don't display the password on the list of Users
    column_exclude_list = ('password',)

    # Don't include the standard password field when creating or editing a User (but see below)
    form_excluded_columns = ('password',)

    # Automatically display human-readable names for the current and available Roles when creating or editing a User
    column_auto_select_related = True

    # Prevent administration of Users unless the currently logged-in user has the "admin" role
    def is_accessible(self):
        return login.current_user.has_role('admin')

    # On the form for creating or editing a User, don't display a field corresponding to the model's password field.
    # There are two reasons for this. First, we want to encrypt the password before storing in the database. Second,
    # we want to use a password field (with the input masked) rather than a regular text field.
    def scaffold_form(self):

        # Start with the standard form as provided by Flask-Admin. We've already told Flask-Admin to exclude the
        # password field from this form.
        form_class = super(UserAdmin, self).scaffold_form()

        # Add a password field, naming it "password2" and labeling it "New Password".
        form_class.password2 = PasswordField('New Password')
        return form_class

    # This callback executes when the user saves changes to a newly-created or edited User -- before the changes are
    # committed to the database.
    def on_model_change(self, form, model, is_created):

        # If the password field isn't blank...
        if len(model.password2):

            # ... then encrypt the new password prior to storing it in the database. If the password field is blank,
            # the existing password in the database will be retained.
            model.password = encrypt_password(model.password2)


class CompetitionAdmin(AdminModelView):

    # Override form field to use Flask-Admin FileUploadField
    form_overrides = {
        'groundtruth': FileUploadField
    }

    # Pass additional parameters to 'path' to FileUploadField constructor
    form_args = {
        'groundtruth': {
            'label': 'Ground truth',
            'base_path': os.path.join(app.config['GROUNDTRUTH_FOLDER']),
            'allow_overwrite': False
        }
    }


@login_required
@app.route('/groundtruth/<filename>')
def get_groundtruth(filename):

    if Competition.query.filter_by(groundtruth=filename).count() == 0:
        abort(404)

    if login.current_user.has_role('admin'):
        return send_from_directory(app.config['GROUNDTRUTH_FOLDER'],
                                   filename)
    else:
        abort(403)


@login_required
@app.route('/submissions/<filename>')
def get_submission(filename):

    submissions = Submission.query.filter_by(filename=filename)

    # make sure the current user is whether admin or the user who actually submitted the file
    if (submissions.count() > 0 and (login.current_user.has_role('admin') or login.current_user.id == submissions.first().user_id)):
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                   filename)
    else:
        abort(403)


class Error(Exception):
    pass


class ParsingError(Error):
    pass
