import re

def clean_answer(options, answer):
    if type(answer) is list:
        answer = answer[0]
    pattern = r"\b[A-D]\b|[A-D](?=\s|:)"
    match = re.search(pattern, answer) 
    if match is None:   
        for option, content in options.items():
            if content in answer:
                return option
        return None
    else:
        return match.group()