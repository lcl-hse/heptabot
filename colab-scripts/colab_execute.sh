source activate heptabot
python
import os
import pickle
import Pyro4
import Pyro4.util
from tqdm import tqdm
Heptamodel = Pyro4.Proxy("PYRONAME:heptabot.heptamodel")
batchify, process_batch, result_to_div = Heptamodel.batchify, Heptamodel.process_batch, Heptamodel.result_to_div
with open("process.pkl", "rb") as inpickle:
    pickledata = pickle.load(inpickle)
task_type, texts = pickledata
prepared_data = {}
for textid in tqdm(texts):
    batches, delims = batchify(texts[textid], task_type)    
    prepared_data[textid] = (batches, delims)
with open("./templates/result.html", "r") as inres:
    outhtml = inres.read()
outhtml = outhtml.replace("{{ which_font }}", "{0}").replace("{{ response }}", "{1}").replace("{{ task_type }}", "{2}")
processed_texts = {}
which_font = "" if task_type == "correction" else "font-family: Ubuntu Mono; letter-spacing: -0.5px;"
task_str = "text" if task_type == "correction" else "sentences"
for textid in tqdm(prepared_data):
    batches, delims = prepared_data[textid]
    processed = []
    if task_type != "correction":
        print("Processing text with ID", textid)
        for batch in tqdm(batches):
            processed.append(process_batch(batch))
    else:
        for batch in batches:
            processed.append(process_batch(batch))
    plist = [item for subl in processed for item in subl] 
    response = result_to_div(texts[textid], plist, delims, task_type)
    proc_html = outhtml.format(which_font, response, task_str)
    with open(os.path.join("output", textid+".html"), "w", encoding="utf-8") as outfile:
        outfile.write(proc_html)