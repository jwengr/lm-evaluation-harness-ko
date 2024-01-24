from datasets import Dataset
from sklearn.metrics import f1_score


def nli_doc_to_text(doc) -> str:
    return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
        doc["premise"],
        doc["hypothesis"].strip()
        + ("" if doc["hypothesis"].strip().endswith(".") else "."),
    )


def macro_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average='macro')
    return fscore
