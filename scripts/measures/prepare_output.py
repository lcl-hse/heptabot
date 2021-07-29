import re
import pickle

with open("process.pkl", "rb") as inpickle:
    texts = pickle.load(inpickle)

for textid in texts:
    task_type = texts[textid]["task_type"]
    with open("./output/" + textid + ".html", "r") as infile:
        with open("./comp_scores/" + task_type + ".res", "w", encoding="utf-8") as outfile:
            outfile.write("\n".join([re.sub(r"^" + task_type + ":?(?: sentence:?)?", "", line, flags=re.IGNORECASE) for line in re.search(r'<div id="resulta".*?>(.*?)</div>', infile.read()).group(1).split("<br>")]))
