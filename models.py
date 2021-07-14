import re
import os
import json
import errant
import torch
import spacy
import Pyro4

import numpy as np
import pandas as pd

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from collections import OrderedDict
from func_timeout import func_set_timeout, FunctionTimedOut
from diff_match_patch import diff_match_patch
from nltk.tokenize import sent_tokenize, word_tokenize
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from transformers import T5TokenizerFast, T5ForConditionalGeneration


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class Heptabot(object):
    def __init__(self):
        pass

    def batchify(self, *args, **kwargs):
        return batchify(*args, **kwargs)

    def process_batch(self, *args, **kwargs):
        return process_batch(*args, **kwargs)

    def result_to_div(self, *args, **kwargs):
        return result_to_div(*args, **kwargs)


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def create_inference_fn():
    global inference
    def inference(input, task="correction", model=tinymodel, num_beams=None,
                  early_stopping=None, no_repeat_ngram_size=None, top_k=None):
        input = tokenizer.encode_plus(f"{task}: "+input, return_tensors="pt",
                                                         max_length=256, truncation=True, padding='max_length')
        out = model.generate(input_ids=input["input_ids"],
                         attention_mask=input["attention_mask"],
                         max_length=256, num_beams=num_beams, early_stopping=early_stopping,
                                     no_repeat_ngram_size=no_repeat_ngram_size, top_k=top_k,
                                     do_sample=True if top_k is not None else False
                                        )
        return tokenizer.decode(out[0], skip_special_tokens=True)


device = 'cuda:0'
if os.environ.get("MODEL_PLACE") == "cpu":
    device = 'cpu'

dmp = diff_match_patch()
nlp = spacy.load("en")
annotator = errant.load('en')

classifier = CatBoostClassifier()
classifier.load_model("./models/classifier/err_type_classifier.cbm")

emb_model = SentenceTransformer('./models/distilbert_stsb_model')

tokenizer = T5TokenizerFast.from_pretrained("./models/T5-small_distilled")
if os.environ.get("MODEL_PLACE") != "tpu":
    tinymodel = T5ForConditionalGeneration.from_pretrained("./models/T5-small_distilled")

    emb_model.to(device)
    tinymodel.to(device)

    create_inference_fn()


def parsify(sent, replace_tab=True):
    if replace_tab:
        sent = sent.replace("\t", " ")

    doc = nlp(sent)
    docsoup = []

    for token in doc:
        if token.is_punct:
            docsoup.append(str(token))
        else:
            info = [str(token.dep_)]
            info += [str(token.pos_)]
            info += [key[key.find("_") + 1:] for key in nlp.vocab.morphology.tag_map[token.tag_].keys() if
                     type(key) == str]
            docsoup.append("_".join(info))

    parsing = " ".join(docsoup)
    ret = "sentence: " + sent + " parsing: " + parsing
    return ret


def batchify_text(text, max_tokens=250):
    def matchlen(instr, max_tokens):
        with suppress_stdout_stderr():
            num_tok = len(tokenizer.encode_plus(instr)[0])
        matches = num_tok < max_tokens
        return num_tok, matches

    def t5ify(instr):
        instr = re.sub("'(.*)'", r"\1", instr)
        instr = re.sub(r"([\s\n\t])+", r"\g<1>", instr)
        instr = re.sub(r"\t", " ", instr)
        instr = re.sub(r"\n", " <br> ", instr)
        return instr

    def maybe_sentencize(orig_parrs, delims, max_tokens):
        new_parrs, new_delims, ntokens_list = [], [], []
        for p, d in zip(orig_parrs, delims):
            num_tok, matches = matchlen(p, max_tokens)
            if matches:
                new_parrs.append(p)
                new_delims.append(d)
                ntokens_list.append(num_tok)
            else:
                sents = sent_tokenize(p)
                for sent in sents:
                    with suppress_stdout_stderr():
                        sent_toks = len(tokenizer.encode_plus(sent)[0])
                    new_parrs.append(sent)
                    new_delims.append(" ")
                    ntokens_list.append(sent_toks)
        return new_parrs, new_delims, ntokens_list

    text = re.sub(r"([\s\n\t])+", r"\g<1>", text.strip())
    preps = re.split(r"(\n)", text) + ["\n"]
    orig_parrs = [preps[i] for i in range(len(preps)) if (i + 1) % 2]
    delims = [preps[i] for i in range(len(preps)) if i % 2]
    orig_parrs, delims, orig_ntokens = maybe_sentencize(orig_parrs, delims, max_tokens)

    curlen = orig_ntokens[0]
    orig = []
    new_delims = []
    cur_or = orig_parrs[0] + delims[0]

    for _i, parrs in enumerate(orig_ntokens[1:]):
        i = _i + 1
        orig_len = orig_ntokens[i]
        delim = delims[i]
        if curlen + orig_len > max_tokens:
            if delims[i]:
                cur_or = cur_or[:-len(delims[i])]
            orig.append(t5ify(cur_or))
            new_delims.append(delim)
            cur_or = orig_parrs[i] + delim
        else:
            cur_or += orig_parrs[i] + delim
            curlen += orig_len

    new_delims.append(delims[-1])
    if new_delims[-1]:
        cur_or = cur_or[:-len(new_delims[-1])]
    orig.append(t5ify(cur_or))

    return orig, new_delims


def batchify(text, task_type):
    if task_type == "correction":
        batches, delims = batchify_text(text)
    else:
        batches = text.split("\n")
        delims = ["<br>"] * len(batches)
        if task_type != "jfleg":
            batches = [parsify(sent) for sent in batches]
    batches = [task_type + ": " + batch for batch in batches]
    return batches, delims


@func_set_timeout(60)
def process_batch(batch):
    global inference
    return re.sub(r"(\W|^)([Ww])ont(\W|$)", r"\1\2on't\3", re.sub(r"\n\s+", r"\n",
                                                                  re.sub(r" +br>", r"\n", inference(batch))))


def spare_spaces(indel, inins):
  if indel is not None:
    ds = re.search(r"^(\s*)(.*?)(\s*)$", indel)
    outdel = ds.group(2)
    add_before = len(ds.group(1))
    add_after = len(ds.group(3))
  else:
    outdel, add_before, add_after = None, 0, 0
  if inins is not None:
    di = re.search(r"^(\s*)(.*?)(\s*)$", inins)
    outins = di.group(2)
  else:
    outins = None
  return outdel, outins, add_before, add_after


def diff_to_ann(diff, classes, original_ann=None):
  if original_ann is not None:
    with open(original_ann, "r") as inann:
      _tdict = {"T": 0, "A": 0, "#": 0}
      for line in [l for l in inann.read().split("\n") if l]:
        s = re.search(r"^([TA#])([0-9]+)\s", line)
        if s:
          _type = s.group(1)
          _id = int(s.group(2))
          if _id > _tdict[_type]:
            _tdict[_type] = _id
    T, A, DASH = _tdict["T"], _tdict["A"], _tdict["#"]
  else:
    T, A, DASH = 0, 0, 0

  class_dict = {
      0: "comp",
      1: "disc",
      2: "punct",
      3: "spell",
      4: "vocab",
      5: "gram"
  }

  ANNS = []
  pos = 0
  _cid = 0

  for k, elem in enumerate(diff):
    mode, change = elem
    if mode == 0:
      pos += len(change[0][1])
    else:
      if len(change) == 2:
        d, i = change[0], change[1]
        if d[0] == 1 and i[0] == -1:
          d, i = i, d
        outdel, outins, add_before, add_after = spare_spaces(d[1], i[1])
        if re.search(r"^\s*$", outdel):
          pos += len(outdel) + add_after
          continue
        pos += add_before
        ANNS.append("T{}\t{} {} {}\t{}".format(T+1, class_dict[classes[_cid]], pos, pos+len(outdel), outdel))
        ANNS.append("#{}\tAnnotatorNotes T{}\t{}".format(DASH+1, T+1, outins))
        pos += len(outdel) + add_after
        T += 1
        DASH += 1
      else:
        m, ch = change[0]
        if m == -1:
          outdel, _, add_before, add_after = spare_spaces(ch, None)
          pos += add_before
          if re.search(r"^\s*$", outdel):
            pos += len(outdel) + add_after
            continue
          ANNS.append("T{}\t{} {} {}\t{}".format(T+1, class_dict[classes[_cid]], pos, pos+len(outdel), outdel))
          ANNS.append("A{}\tDelete T{}".format(A+1, T+1))
          pos += len(outdel) + add_after
          T += 1
          A += 1
        else:
          _, outins, _, _ = spare_spaces(None, ch)
          if pos == 0:
            rs = re.search(r"^(\s*)(.*?)(?:[^-'\w]*)(?:\s|$)", diff[1][1][0][1])
            add_before = len(rs.group(1))
            pseudodel = rs.group(2)
            ANNS.append("T{}\t{} {} {}\t{}".format(T+1, class_dict[classes[_cid]], add_before, add_before+len(pseudodel), pseudodel))
            ANNS.append("#{}\tAnnotatorNotes T{}\t{}".format(DASH+1, T+1, outins))
            T += 1
            DASH += 1
          else:
            rs = re.search(r"(?:\s|^)(\S*?)([^-'\w\s]*)(\s*)$", diff[k-1][1][0][1])
            pseudodel = rs.group(1)
            punct = rs.group(2)
            len_diff = len(diff[k-1][1][0][1])
            len_punct = len(punct)
            add_after = len(rs.group(3))
            add_before = len_diff - len(pseudodel) - len_punct - add_after
            ANNS.append("T{}\t{} {} {}\t{}".format(T+1, class_dict[classes[_cid]], pos - len(pseudodel) - len_punct - add_after, pos - len_punct - add_after, pseudodel + punct))
            ANNS.append("#{}\tAnnotatorNotes T{}\t{}".format(DASH+1, T+1, outins))
            T += 1
            DASH += 1
      _cid += 1
  return "\n".join(ANNS)


def merge_results(batches, delims):
    delims = delims[:len(delims) - 1] + [""]
    outs = [p + d for p, d in zip(batches, delims)]
    return "".join(outs)


def errant_process(origs, corrs, annotator):
    ori = annotator.parse(origs, tokenise=True)
    cor = annotator.parse(corrs, tokenise=True)
    alignment = annotator.align(ori, cor)
    edits = annotator.merge(alignment)
    edit_list = [annotator.classify(e) for e in edits]
    return ori, cor, edit_list


def predict_error_class(errors, corrections, model, sentence_embedder, tokenizer):
    def is_capitalised(instr):
        return int(instr.istitle())

    def is_punct(instr):
        if re.search(r"^\W+$", instr, flags=re.DOTALL):
            return 1
        return 0

    def endswith_punct(instr):
        if re.search(r"\W$", instr, flags=re.DOTALL):
            return 1
        return 0

    def is_num(instr):
        if re.search(r"^[0-9]+$", instr, flags=re.DOTALL):
            return 1
        return 0

    if len(errors) != len(corrections):
        raise IndexError("Lengths of error and correction lists do not match")

    pd_dict = OrderedDict()

    pd_dict["orig_str_len"] = [len(e) for e in errors]
    pd_dict["corr_str_len"] = [len(c) for c in corrections]

    pd_dict["orig_str_tok"] = [len(tokenizer(e)) for e in errors]
    pd_dict["corr_str_tok"] = [len(tokenizer(c)) for c in corrections]

    pd_dict["orig_str_title"] = [is_capitalised(e) for e in errors]
    pd_dict["corr_str_title"] = [is_capitalised(c) for c in corrections]

    pd_dict["orig_str_punct"] = [is_punct(e) for e in errors]
    pd_dict["corr_str_punct"] = [is_punct(c) for c in corrections]

    pd_dict["orig_str_punct_end"] = [endswith_punct(e) for e in errors]
    pd_dict["corr_str_punct_end"] = [endswith_punct(c) for c in corrections]

    pd_dict["orig_str_num"] = [is_num(e) for e in errors]
    pd_dict["corr_str_num"] = [is_num(c) for c in corrections]

    error_df = pd.DataFrame(pd_dict)

    embeddings = sentence_embedder.encode(errors + corrections)

    err_embs = embeddings[:len(errors)]
    corr_embs = embeddings[len(errors):]

    embdiff_pd = pd.DataFrame([o - s for o, s in zip(err_embs, corr_embs)])
    embdiff_pd.columns = ["vec" + str(v) for v in embdiff_pd.columns]

    error_df = pd.concat([error_df, embdiff_pd], axis=1)

    preds = model.predict(error_df)

    preds = list(np.ndarray.flatten(preds))
    return preds


def diff_from_errant(orig, corr, patch_list):
    """
    Applies correction to an individual text entry
    :param text: string containing original uncorrected text
    :param patch_list: list containing patches in [start, end, correction] notation
    :return: original text (str), corrected text (str), number of corrections (int)
    """
    I = 0
    outlist = []
    trail = ""
    lastc, lasto = "", ""

    for patch in patch_list:
        start, end, original, correction = patch.o_start, patch.o_end, patch.o_str, patch.c_str
        if start - I:
            zstr = trail + "".join(str(orig[i]) + orig[i].whitespace_ for i in range(I, start))
            if orig[start - 1].whitespace_ != corr[patch.c_start - 1].whitespace_:
                zstr = zstr[:len(zstr) - len(orig[start - 1].whitespace_)]
            outlist.append((0, zstr))
            lastc = zstr
            lasto = zstr
        I = end
        ows, cws = orig[patch.o_end - 1].whitespace_, corr[patch.c_end - 1].whitespace_
        trail = ""
        for _o, _c in zip(ows[::-1], cws[::-1]):
            if _o == _c:
                trail += _o
            else:
                break
        trail = trail[::-1]
        ows = ows[:len(ows)-len(trail)] if ows[:len(ows)-len(trail)] else ""
        cws = cws[:len(cws)-len(trail)] if cws[:len(cws)-len(trail)] else ""
        if original or ows:
            ws = orig[patch.o_start - 1].whitespace_
            s = ""
            if start != end:
                s = "" if lasto.endswith(ws) else ws
            lasto = s + original + ows
            outlist.append((-1, lasto))
        if correction or cws:
            ws = corr[patch.c_start - 1].whitespace_
            s = ""
            if patch.c_start != patch.c_end:
                s = "" if lastc.endswith(ws) else ws
            lastc = s + correction + cws
            outlist.append((1, lastc))

    if I < len(orig) - 1:
        outlist.append((0, trail + "".join(str(orig[i]) + orig[i].whitespace_ for i in range(I, len(orig)))))

    return outlist


def merge_diff(difflist):
    outlist = []
    errlist = []

    prev0, prev1 = False, False
    z = [0, ""]
    e = [-1, ""]
    c = [1, ""]

    for t, dstr in difflist:
        if t == 0:
            if prev1:
                outlist.append(tuple(e))
                outlist.append(tuple(c))
                errlist.append([e[1], c[1]])
            z[1] += dstr
            if not prev0:
                prev0, prev1 = True, False
                e = [-1, ""]
                c = [1, ""]
        else:
            if prev0:
                outlist.append(tuple(z))
            if t == 1:
                c[1] += dstr
            else:
                e[1] += dstr
            if not prev1:
                prev0, prev1 = False, True
                z = [0, ""]

    if prev0:
        outlist.append(tuple(z))
    if prev1:
        outlist.append(tuple(e))
        outlist.append(tuple(c))
        errlist.append([e[1], c[1]])

    return outlist, errlist


def despacify_diff(diff):
    diff = [list(t) for t in diff if t[1]]
    for i, elem in enumerate(diff):
        check = False
        t, _ = elem
        if i:
            if t == 1 and diff[i - 1][0] == 0:
                check = True
        if i != len(diff) - 1:
            if t == -1 and diff[i + 1][0] == 0:
                check = True
        if check:
            if i != len(diff) - 1:
                c = re.search(r"^( +)", diff[i + 1][1])
                if c:
                    diff[i][1] = diff[i][1] + diff[i + 1][1][:c.span(1)[1]]
                    diff[i + 1][1] = diff[i + 1][1][c.span(1)[1]:]
                    continue
            if i:
                c = re.search(r"( +)$", diff[i - 1][1])
                if c:
                    diff[i][1] = diff[i - 1][1][c.span(1)[0]:] + diff[i][1]
                    diff[i - 1][1] = diff[i - 1][1][:c.span(1)[0]]
                    continue
    return [tuple(t) for t in diff]


def groupify_diff(diff):
    output = []
    current = [diff[0]]

    for i in range(1, len(diff)):
        if abs(diff[i][0]) != abs(diff[i - 1][0]):
            output.append((abs(diff[i - 1][0]), current))
            current = []
        current.append(diff[i])

    output.append((abs(diff[-1][0]), current))

    return output


def diff_prettyHtml(diff, classes):
    """Convert diff and classes lists into a HTML report.
    Args:
      diff: List of diff tuples.
      classes: List of error classes.
    Returns:
      HTML representation.
    """

    ERROR = 1
    EQUAL = 0

    class_dict = {
        0: "comp",
        1: "disc",
        2: "punct",
        3: "spell",
        4: "vocab",
        5: "gram"
    }

    desc_dict = {
        0: "Complex grammar error",
        1: "Discourse error",
        2: "Punctuation error",
        3: "Spelling error",
        4: "Vocabulary error",
        5: "Word-level grammar error"
    }

    html = []
    I = -1
    changed = False
    for (op, data) in diff:
        waserror = False
        if abs(op) == ERROR:
            I += 1
            cur_class = class_dict[classes[I]]
            curr_desc = desc_dict[classes[I]]
            ret = '<div style="display: inline;" onmouseover="showcomment(this, event);" onmouseleave="hidecomment(this);">'
            for elem in data:
                if elem[0] == -1:
                    waserror = True
                    err = (elem[1].replace("&", "&amp;").replace("<", "&lt;")
                           .replace(">", "&gt;").replace("\n", "<br>"))
                    ret += '<del class="hidden {}" style="cursor: pointer;" onclick="showhide(this);">{}</del>'.format(
                        cur_class, err)
                    ret += '<div class="{} error-hider" onclick="showhide(this);"></div>'.format(cur_class)
                if elem[0] == 1:
                    corr = (elem[1].replace("&", "&amp;").replace("<", "&lt;")
                            .replace(">", "&gt;").replace("\n", "<br>"))
                    ret += '<ins class="{}"'.format(cur_class)
                    if waserror:
                        ret += ' style="cursor: pointer;" onclick="showhide(this);"'
                    ret += '>{}</ins>'.format(corr)
            ret += '<hgroup class="{} error-type" style="left: 709.15px; visibility: hidden; top: 70.5px; --left-pos:NaNpx;"><span>{}</span></hgroup></div>'.format(
                cur_class, curr_desc)
            html.append(ret)
        elif abs(op) == EQUAL:
            text = (data[0][1].replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;").replace("\n", "<br>"))
            html.append("<span>{}</span>".format(text))
    return "".join(html)


@func_set_timeout(30)
def result_to_div(text, response_obj, delims, task_type, maybe_to_ann=False, original_ann=None):
    if maybe_to_ann:
        origs = re.sub(r"[\n\t]", " ", text)
    else:
        origs = re.sub(r"(\s)+", r"\g<1>", text)
    corrs = merge_results(response_obj, delims)
    if maybe_to_ann:
        corrs = re.sub(r"[\n\t]", " ", corrs)
    if task_type == "correction":
        diff = diff_from_errant(*errant_process(origs, corrs, annotator))
        diff, errlist = merge_diff(diff)
        errors = [e[0] for e in errlist]
        corrections = [e[1] for e in errlist]
        classes = predict_error_class(errors=errors, corrections=corrections,
                                      model=classifier, sentence_embedder=emb_model,
                                      tokenizer=word_tokenize)
        diff = groupify_diff(despacify_diff(diff))
        if maybe_to_ann:
            res = diff_to_ann(diff, classes, original_ann=original_ann)
        else:
            res = diff_prettyHtml(diff, classes)
    return res


def main():
    Pyro4.Daemon.serveSimple(
        {
            Heptabot: "heptabot.heptamodel"
        },
        ns=True)


if __name__ == "__main__":
    main()
