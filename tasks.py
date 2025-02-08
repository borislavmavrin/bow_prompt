import json


def get_bbh_causal_judgement():
    return json.load(open("data/tasks/causal_judgement.json"))["examples"]


def get_boolq():
    tasks = list()
    rows = [json.loads(row.strip()) for row in open("data/tasks/boolq_dev.jsonl").readlines()]
    for row in rows:
        input = f"Question: {row['question']}?"
        input += f"\nPassage: {row['passage']}"
        input += f"\nAnswer options:\nyes\nno"
        target = "yes" if row["answer"] else "no"
        tasks.append(dict(input=input, target=target))
    return tasks
