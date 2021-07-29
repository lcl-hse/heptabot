import os
import pickle
import Pyro4
import Pyro4.util
from tqdm import tqdm

Heptamodel = Pyro4.Proxy("PYRONAME:heptabot.heptamodel")
batchify = Heptamodel.batchify

with open("./raw/process_texts.pkl", "rb") as inpickle:
    texts = pickle.load(inpickle)

print("Preparing texts for TPU model inference")
with open("./raw/raw_input.txt", "w") as outtext:
    prepared_data = {}
    _cnt = 0
    for textid in tqdm(texts, position=0, leave=True):
        batches, delims = batchify(texts[textid]["text"], texts[textid]["task_type"], model_batches=1)
        prepared_data[textid] = {"strings": [], "delims": delims}
        for el in batches:
            outtext.write(el[0]+"\n")
            prepared_data[textid]["strings"].append(_cnt)
            _cnt += 1

with open("./raw/batchified.pkl", "wb") as outpickle:
    pickle.dump(prepared_data, outpickle)
