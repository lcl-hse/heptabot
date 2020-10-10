import os
import time

from flask import Flask, Markup, Response, request, stream_with_context, redirect, url_for, render_template, jsonify
from werkzeug.exceptions import HTTPException

from models import batchify, process_batch, result_to_div

class InputOverflow(Exception):
    """Error for exceptionally massive requests."""
    pass


app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('heptabot'))

@app.route('/heptabot', methods=['GET', 'POST'])
def heptabot():
    global batches, delims, processed, text, task_type
    
    if request.method == 'POST':
        text = request.form['text']
        task_type = request.form['task-type']
        
        with open("generated.txt", "w", encoding="utf-8") as infile:
            infile.write(text)
        
        processed = []
        batches, delims = batchify(text, task_type)
        
        if len(batches) > 50:
            raise InputOverflow(task_type)
        
        return render_template('processing.html', total=str(len(batches)))
    return render_template('index.html')

@app.route('/slow')
def slow():
    global batches, delims, processed, text, task_type, response
    
    for batch in batches:
        processed.append(process_batch(batch))
    
    plist = [item for subl in processed for item in subl] 
    response = Markup(result_to_div(text, plist, delims, task_type))
    
    return jsonify("Done.")

@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    error_obj = {}
    
    if isinstance(e, InputOverflow):
        if str(e) == "correction":
            error_obj["header"] = "It seems that your entry has more than 25,000 words."
        else:
            error_obj["header"] = "It seems that your entry has more than 100 sentences."
        error_obj["str1"] = "In order to maintain server resources and stable uptime, we limit the amounts of data that can be processed via our Web interface."
        error_obj["str2"] = Markup('You can process your data in our <a href="">Colab notebook</a> instead. <a href="/download">Click here</a> to download your data.')

    else:
        error_obj["header"] = "Seems like a runtime error has occurred. Here's the info:"
        error_obj["str1"] = Markup("<b>{}:</b> {}".format(e.__class__.__name__, str(e)))
        error_obj["str2"] = Markup('You can report the error via our <a href="https://forms.gle/RpFdgLN92L4KQ3DMA">Google Form</a>.')
    
    return render_template("error.html", error_obj=error_obj), 500

@app.route('/progress')
def progress():
    def generate():
        global batches, processed
        current = len(processed)
        num_batches = len(batches)
        while current != num_batches:
            current = len(processed)
            num_batches = len(batches)
            yield "data:" + str(num_batches) + " " +  str(current) + "\n\n"
            time.sleep(0.1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download')
def downloadFile():
    path = "generated.txt"
    return send_file(path, as_attachment=True)

@app.route('/result')
def result():
    global task_type, response
    task = "text" if task_type == "correction" else "sentences"
    which_font = "" if task_type == "correction" else "font-family: Ubuntu Mono; letter-spacing: -0.5px;"
    return render_template('result.html', response=response, task_type=task, which_font=which_font)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="8080")

