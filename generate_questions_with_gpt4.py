import datasets
import jsonlines
import openai

from utils import load_secrets

secrets = load_secrets()
openai.api_key = secrets["openai_api_key"]
openai.organization = secrets["openai_organization"]


def generate_questions_given_abstract(title, abstract):
    COMPLETIONS_MODEL = "gpt-4"
    CHAT_COMPLETIONS_API_PARAMS = {
        "temperature": 0.7,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
    }

    prompt = f"""

    Below is a title and abstract of a scientific research paper.

    Title: {title}
    Abstract: {abstract}

    List all questions that a curious reader might have after reading this abstract. Only list questions that you believe can be answered in the full text of the paper. For each question, provide a short phrase (no more than three words) directly from the abstract that is most closely related to each question.

    Questions:

    """

    try:
        response = openai.ChatCompletion.create(
            **CHAT_COMPLETIONS_API_PARAMS,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant designed to ask questions about a scientific paper given its abstract.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "FAILED"


def main():
    dt = datasets.load_dataset("qasper")

    # Load previously processed paper ids so we don't recompute (expensive!)
    with jsonlines.open("gpt4_gen_qasper_qs.jsonl") as reader:
        ids = {paper["id"] for paper in reader}

    for split, split_dt in dt.items():
        num_examples = 100
        for i, x in enumerate(split_dt):
            if i >= num_examples:
                return

            paper_id = x["id"]
            if paper_id in ids:
                print(f"{paper_id} already processed. Skipping..")
                continue

            title, abstract = x["title"], x["abstract"]
            questions = generate_questions_given_abstract(title, abstract)

            with jsonlines.open("gpt4_gen_qasper_qs.jsonl", mode="a") as writer:
                writer.write(
                    {"id": paper_id, "title": title, "abstract": abstract, "questions": questions}
                )


if __name__ == "__main__":
    main()
