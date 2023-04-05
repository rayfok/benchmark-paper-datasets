import re
from collections import defaultdict

import jsonlines


def format_questions():
    with jsonlines.open("gpt4_gen_qasper_qs.jsonl") as reader:
        paper_questions = {paper["id"]: paper["questions"] for paper in reader}

    formatted = defaultdict(list)
    for paper_id, questions in paper_questions.items():
        if questions == "FAILED":
            formatted[paper_id] = []
            continue

        try:
            questions = questions.split("\n\n")
        except ValueError:
            print(questions)
            continue

        for q_with_excerpt in questions:
            try:
                question, excerpt = q_with_excerpt.split("\n")
            except ValueError:
                ## handle excerpt in parentheses e.g., <question> (<excerpt)
                question = re.sub(r"\(.*\)", "", q_with_excerpt)
                excerpt = re.search(r"\((.*?)\)", q_with_excerpt).group(1)
                if not excerpt:
                    print(
                        f"[WARNING] Excerpt not found in parentheses: {q_with_excerpt}."
                        f"Paper: {paper_id}"
                    )
                    continue


            excerpt = excerpt.strip()
            excerpt = re.sub(r"Phrase: ", "", excerpt, flags=re.IGNORECASE)
            excerpt = re.sub(r"Related ", "", excerpt, flags=re.IGNORECASE)
            excerpt = re.sub('"', "", excerpt)
            excerpt = re.sub(r"^- ", "", excerpt)

            question = question.strip()
            question = re.sub(r"^\d+\.\s*", "", question)

            formatted[paper_id].append((question, excerpt))

    with jsonlines.open("gpt4_gen_qasper_qs_formatted.jsonl", mode="w") as writer:
        for paper_id, questions in formatted.items():
            writer.write({"id": paper_id, "questions": questions})


if __name__ == "__main__":
    format_questions()
