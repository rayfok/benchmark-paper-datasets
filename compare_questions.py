import datasets
import jsonlines


def compare_questions():
    papers = {}

    ## Load GPT4-generated questions
    with jsonlines.open("gpt4_gen_qasper_qs_formatted.jsonl") as reader:
        gpt4_questions = {}
        for paper in reader:
            id, questions = paper["id"], paper["questions"]
            gpt4_questions[id] = [question for question, excerpt in questions]

    ## Load QASPER- (NLP crowd) generated questions
    dt = datasets.load_dataset("qasper")
    qasper_questions = {}
    for split_dt in dt.values():
        for x in split_dt:
            id = x["id"]
            if id in gpt4_questions:
                qasper_questions[id] = x["qas"]["question"]
                papers[id] = {"title": x["title"], "abstract": x["abstract"]}

    for paper_id in gpt4_questions.keys():
        title = papers[paper_id]["title"]
        abstract = papers[paper_id]["abstract"]

        print(paper_id)
        print(f"Title: {title}\nAbstract: {abstract}")

        print("=== GPT4 questions ===")
        for i, q in enumerate(gpt4_questions[paper_id]):
            print(f"{i+1}. {q}")
        print()

        print("=== QASPER questions ===")
        for i, q in enumerate(qasper_questions[paper_id]):
            print(f"{i+1}. {q}")
        print()

        input()


if __name__ == "__main__":
    compare_questions()
