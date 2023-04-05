import json
from collections import defaultdict
from pathlib import Path

import datasets
import jsonlines
import numpy as np
import openai

from utils import load_secrets

secrets = load_secrets()
openai.api_key = secrets["openai_api_key"]
openai.organization = secrets["openai_organization"]

RESULTS_OUTPUT_FILEPATH = "gpt4_qasper_test_v2.jsonl"


def explode_questions(batch: dict):
    exploded = {
        "id": [],
        "title": [],
        "abstract": [],
        "question": [],
        "answer": [],
        "full_text": [],
    }

    for i in range(len(batch["id"])):
        title = batch["title"][i]
        abstract = batch["abstract"][i]
        full_text = batch["full_text"][i]
        example_id = batch["id"][i]
        qas = batch["qas"][i]
        for qid, question, answer_data in zip(qas["question_id"], qas["question"], qas["answers"]):
            computed_answers = set()
            for answer in answer_data["answer"]:
                # if any(e.startswith('FLOAT SELECTED') for e in answer['evidence']):
                #     continue
                if answer["unanswerable"]:
                    computed_answers.add("Unanswerable")
                elif answer["yes_no"] is not None:
                    computed_answers.add("Yes" if answer["yes_no"] else "No")
                elif answer["extractive_spans"]:
                    computed_answers.add(" ... ".join(answer["extractive_spans"]))
                else:
                    computed_answers.add(answer["free_form_answer"])

            if not computed_answers:
                continue

            exploded["id"].append(f"{example_id}.{qid}")
            exploded["title"].append(title)
            exploded["abstract"].append(abstract)
            exploded["question"].append(question)
            exploded["answer"].append(sorted(computed_answers))
            exploded["full_text"].append(full_text)

    return exploded


def clean_up_full_text(sample: dict):
    full_text = f"{sample['title']}\n\n\nAbstract\n\n{sample['abstract']}\n\n\n"
    for sec_name, paras in zip(
        sample["full_text"]["section_name"], sample["full_text"]["paragraphs"]
    ):
        if len(paras) == 0:
            continue

        if sec_name:
            sec_name = sec_name.split(":::")[-1].strip()
            full_text += f"{sec_name}\n\n"
        for para in paras:
            full_text += f"{para}\n\n"
        full_text += "\n\n"

    sample["body"] = full_text.strip()
    return sample


def query_paper(sample: dict, cache: dict = {}):
    sample_id = sample["id"]

    if sample_id in cache:
        print(f"{sample_id} already processed")
        return

    COMPLETIONS_MODEL = "gpt-4"
    CHAT_COMPLETIONS_API_PARAMS = {
        "temperature": 0.5,
        "max_tokens": 128,
        "model": COMPLETIONS_MODEL,
    }
    MAX_BODY_WORDS = 5000
    body = " ".join(sample["body"].split(" ")[:MAX_BODY_WORDS])
    prompt = f"""{body}

        Question: {sample["question"]}

        Answer:
    """

    try:
        response = openai.ChatCompletion.create(
            **CHAT_COMPLETIONS_API_PARAMS,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant designed to answer questions about a paper. Use the text of the paper, which is provided in triple quotes, to answer the question given by the user. Papers are truncated to a maximum of {MAX_BODY_WORDS} words. All the answers should be short (1 sentence maximum). The paper might not contain the answer; respond with `Unanswerable` if that is the case.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        sample["gpt4_answer"] = response.choices[0].message.content.strip()
    except Exception as e:
        sample["gpt4_answer"] = "FAILED"
        print("FAILED:", e)
    del sample["body"]
    with jsonlines.open(RESULTS_OUTPUT_FILEPATH, mode="a") as writer:
        writer.write(dict(sample))

    cache[sample_id] = sample


def get_qasper_stats(dt):
    num_questions_per_paper = defaultdict(int)
    num_questions_per_writer = defaultdict(int)
    num_questions_per_writer_per_paper = defaultdict(int)
    for split, split_dt in dt.items():
        for x in split_dt:
            paper_id = x["id"]
            qas = x["qas"]
            question_writers = qas["question_writer"]

            num_questions_per_paper[paper_id] += len(qas["question_id"])
            for writer_id in question_writers:
                num_questions_per_writer[writer_id] += 1
                num_questions_per_writer_per_paper[(paper_id, writer_id)] += 1

            for q in qas["question"]:
                print(q)
            print()

    return {
        "questions_per_paper": np.mean(list(num_questions_per_paper.values())),
        "questions_per_writer": np.mean(list(num_questions_per_writer.values())),
        "questions_per_writer_per_paper": np.mean(
            list(num_questions_per_writer_per_paper.values())
        ),
    }


def main():
    dt = datasets.load_dataset("qasper")

    ## Compare gpt4 generated questions with crowd-generated questions in qasper
    with jsonlines.open("gpt4_gen_qasper_qs.jsonl") as reader:
        ids = {paper["id"] for paper in reader}
    for split_dt in dt.values():
        for x in split_dt:
            if x["id"] in ids:
                print(f"[{x['id']}] {x['title']}")
                crowd_qs = x["qas"]["question"]
                for q in crowd_qs:
                    print(q)
                print()
    return

    stats = get_qasper_stats(dt)
    print(json.dumps(stats, indent=2))

    return

    # Load previous results if they exist
    cache = {}
    if Path(RESULTS_OUTPUT_FILEPATH).is_file():
        with jsonlines.open(RESULTS_OUTPUT_FILEPATH) as reader:
            for row in reader:
                cache[row["id"]] = row

    splits = {}
    total = 0
    for split in dt.keys():
        splits[split] = dt[split].map(
            explode_questions,
            batched=True,
            batch_size=100,
            remove_columns=dt[split].column_names,
            load_from_cache_file=False,
        )
        total += len(splits[split])
    print(splits)
    print(total)

    for split in splits.keys():
        splits[split] = splits[split].map(
            clean_up_full_text,
            load_from_cache_file=False,
            remove_columns=["full_text", "title", "abstract"],
        )

    for split in splits.keys():
        splits[split] = splits[split].map(
            lambda x: {"num_words": len(x["body"].split(" "))},
            load_from_cache_file=False,
        )

    ## Print questions in all splits
    # for split_dt in splits.values():
    #     split_dt.map(lambda x: print(x["question"]))

    return

    selected_dataset = splits["test"]
    selected_dataset.map(query_paper, fn_kwargs={"cache": cache})

    for id, row in cache.items():
        if row["gpt4_answer"] == "FAILED":
            failed_row = selected_dataset.filter(lambda x: x["id"] == id)[0]
            query_paper(failed_row, cache={})


if __name__ == "__main__":
    main()
