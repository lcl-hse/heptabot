import os
import time

from flask import Flask, Markup, Response, request, stream_with_context, redirect, url_for, render_template
from werkzeug.exceptions import HTTPException

from models import batchify, process_batch, result_to_div

class InputOverflow(Exception):
    """Error for exceptionally massive requests."""
    pass

def page_generator(request):
    global batches, processed
    text = request.form['text']
    task_type = request.form['task-type']
    
    with open("generated.txt", "w", encoding="utf-8") as infile:
        infile.write(text)
    
    processed = []
    batches, delims = batchify(text, task_type)
    
    if len(batches) > 50:
        raise InputOverflow(task_type)
    
    yield render_template('processing.html', total=str(len(batches)))
    
    for batch in batches:
        processed.append(process_batch(batch))
    
    plist = [item for subl in processed for item in subl] 
    response = Markup(result_to_div(text, plist, delims, task_type))
    
    yield redirect(url_for('result'), response=response, task=task_type)

generator_obj = None

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('heptabot'))

@app.route('/heptabot', methods=['GET', 'POST'])
def heptabot():
    global generator_obj
    if request.method == 'POST':
        generator_obj = generator_obj or page_generator(request)
        return Response(stream_with_context(next(generator_obj)))
    return render_template('index.html')

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
        percentage = len(processed) / len(batches)
        while x <= 100.0:
            yield "data:" + str(percentage) + " " +  str(current) + "\n\n"
            x = x + 10
            time.sleep(0.05)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download')
def downloadFile():
    path = "generated.txt"
    return send_file(path, as_attachment=True)

@app.route('/result')
def result(response, task):
    task_type = "text" if task == "correction" else "sentences"
    which_font = "" if task == "correction" else "font-family: Ubuntu Mono;"
    return render_template('result.html', response=response, task_type=task_type, which_font=which_font)


if __name__ == "__main__":
    app.run(debug=True)
