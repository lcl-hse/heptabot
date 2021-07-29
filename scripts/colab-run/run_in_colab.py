import os
import pickle
import Pyro4
import Pyro4.util
from tqdm import tqdm

Heptamodel = Pyro4.Proxy("PYRONAME:heptabot.heptamodel")
batchify, process_batch, result_to_div = Heptamodel.batchify, Heptamodel.process_batch, Heptamodel.result_to_div

with open("process.pkl", "rb") as inpickle:
    texts = pickle.load(inpickle)

prepared_data = {}
for textid in texts:
    batches, delims = batchify(texts[textid]["text"], texts[textid]["task_type"])    
    prepared_data[textid] = (texts[textid]["task_type"], batches, delims)

with open("./templates/result.html", "r") as inres:
    outhtml = inres.read()
outhtml = outhtml.replace("{{ which_font }}", "{0}").replace("{{ response }}", "{1}").replace("{{ task_type }}", "{2}")

for textid in tqdm(prepared_data, position=0, leave=True):
    task_type, batches, delims = prepared_data[textid]
    which_font = "" if task_type == "correction" else "font-family: Ubuntu Mono; letter-spacing: -0.5px;"
    task_str = "text" if task_type == "correction" else "sentences"
    processed = []

    if task_type != "correction":
        print("\nProcessing text with ID", textid)
        for batch in tqdm(batches, position=0, leave=True):
            processed.append(process_batch(batch))
    else:
        for batch in batches:
            processed.append(process_batch(batch))
    response = result_to_div(texts[textid]["text"], processed, delims, task_type)

    proc_html = outhtml.format(which_font, response, task_str)
    with open(os.path.join("output", textid + ".html"), "w", encoding="utf-8") as outfile:
        outfile.write(proc_html)
