import os
import pickle

texts = {}

for task_type in ("correction", "jfleg", "conll", "bea"):
  if task_type in os.listdir("input"):
    for tfid in os.listdir(os.path.join("input", task_type)):
      textid = os.path.splitext(tfid)[0]
      with open(os.path.join("input", task_type, tfid), "r", encoding="utf-8") as infile:
        texts[textid] = {"task_type": task_type, "text": infile.read().strip()}

with open("process.pkl", "wb") as outpickle:
  pickle.dump(texts, outpickle)
