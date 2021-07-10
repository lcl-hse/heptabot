import re
import os
import pickle
import Pyro4
import Pyro4.util
from tqdm import tqdm

def unprocess(instr):
    instr = re.sub(r" +<br> ?", r"\n", instr.replace(chr(8263) + " ", "<"))
    return instr

Heptamodel = Pyro4.Proxy("PYRONAME:heptabot.heptamodel")
result_to_div = Heptamodel.result_to_div

with open("./raw/process_texts.pkl", "rb") as inpickle:
    pickledata = pickle.load(inpickle)
task_type, texts = pickledata

with open("./raw/batchified.pkl", "rb") as inpickle:
    prepared_data = pickle.load(inpickle)

CHECKPOINT_STEP = os.environ["CHECKPOINT_STEP"]
with open("./raw/raw_output.txt-" + CHECKPOINT_STEP, "r") as intxt:
    processed_texts = [t for t in intxt.read().split("\n") if t]

with open("./templates/result.html", "r") as inres:
    outhtml = inres.read()
outhtml = outhtml.replace("{{ which_font }}", "{0}").replace("{{ response }}", "{1}").replace("{{ task_type }}", "{2}")

which_font = "" if task_type == "correction" else "font-family: Ubuntu Mono; letter-spacing: -0.5px;"
task_str = "text" if task_type == "correction" else "sentences"

print("Processing TPU model outputs")
for textid in tqdm(prepared_data):
    strings, delims = prepared_data[textid]["strings"], prepared_data[textid]["delims"]
    plist = [unprocess(processed_texts[sid]) for sid in strings]
    response = result_to_div(texts[textid], plist, delims, task_type)
    
    proc_html = outhtml.format(which_font, response, task_str)
    with open(os.path.join("output", textid+".html"), "w", encoding="utf-8") as outfile:
        outfile.write(proc_html)
