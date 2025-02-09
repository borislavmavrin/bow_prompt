def create_few_shot_examples(tasks):
    messages = list()
    for task in tasks:
        messages.extend(
            [
                dict(role="user", content=task["input"]),
                dict(role="assistant", content=task["target"])
            ]
        )
    return messages
