import os
import json
import argparse
from pathlib import Path
import re

import ollama

def find_text_and_json_files(input_dir, filter_paths):
    """
    Traverse the input directory to find text files and their corresponding JSON files
    that match the filter_paths.
    """
    files_to_process = {}
    filter_paths = list(filter_paths)

    # filter_stems = [Path(path).parent.name + "/" + Path(path).name for path in filter_paths]

    folder = input_dir.split("/")[-1]
    base_folder = '/'.join(input_dir.split("/")[:-1])

    cnt_absent, txt_cnt = 0, 0
    for f in filter_paths:
        json_file = os.path.join(base_folder, f)

        f_directory = os.path.dirname(json_file)

        for root, _, files in os.walk(f_directory):
            text_file = None
            for filename in files:
                if filename.endswith("text.en.txt"):
                    text_file = os.path.join(root, filename)
                    break  # Assume only one text file per directory

                if filename.endswith("text.txt") and all(not f.endswith("text.en.txt") for f in files):
                    text_file = os.path.join(root, filename)
                    break  # Assume only one text file per directory

            if text_file:
                txt_cnt += 1

            files_to_process[json_file.replace(input_dir, folder)] = (text_file, json_file)

        if not os.path.exists(json_file):
            cnt_absent += 1

    print(f"Found {len(filter_paths)} filter paths, {cnt_absent} of them are absent.")
    print(f"Recorded {len(files_to_process.keys())} json files with {txt_cnt} text files:")

    return files_to_process


def load_context(text_file_path):
    """
    Read the content of the text file as context.
    """
    with open(text_file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_qa_pairs(json_file_path):
    """
    Load question-answer pairs from the JSON file.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_json(input_string):
    """
    Extracts a JSON substring from a given input string.
    The function searches for content between the first { and the last }.

    Args:
        input_string (str): String potentially containing JSON

    Returns:
        dict: Parsed JSON object or None if extraction fails
    """
    # Find the substring that starts with { and ends with }
    match = re.search(r'(\{.*\})', input_string, re.DOTALL)

    if match:
        json_string = match.group(1)
        try:
            # Parse the extracted JSON string
            json_object = json.loads(json_string)
            score = int(json_object.get("score", 0))
            explanation = json_object.get("explanation", json_string)
            return {"score": score, "explanation": explanation}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {"score": 0, "explanation": json_string}
    else:
        print("No JSON object found in the input string")
        match = re.search(r'\b([0-9]{1,3})\b', input_string)
        if match:
            score = int(match.group(1))
            return {"score": score, "explanation": input_string}
        return {"score": 0, "explanation": input_string}


def evaluate_answers(model, text_fragments, qa_pairs):
    """
    For each question-answer pair, prompt the model and extract the score.
    """
    sys = (
        f"You are a fair teacher who grades students' answers. Evaluate the quality of the *Answer* specifically in response to the *Question*, "
        f"considering the *Context* provided."
        f"Format your entire response as a single JSON object containing 'score' (an integer between 0 and 100, where 100 is best) "
        f"and 'explanation' (a string briefly justifying the score). "
    )

    scores, expls = [], []
    for idx, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        # answer = qa.get("reference-answers", [])[0] if len(answer) == 0 and "answer" not in qa.keys() else answer

        # print(question, answer, text_fragments)

        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"And given the following context:\n{text_fragments}\n"
        )

        try:
            response = ollama.generate(model=model, prompt=prompt, options={'num_ctx': 40950}, system=sys)
            response_text = response['response']  # Assuming 'response' key holds the generated text

            result_dict = extract_json(response_text)

            print(f"\nProcessed QA pair {idx + 1}/{len(qa_pairs)}: Score =\t{result_dict['score']}\t points")
            print(f"\tExplanation: {result_dict['explanation']}")
            print(f"\tQuestion: {question}\n\tAnswer: {answer}")

            scores.append(result_dict['score'])
            expls.append(result_dict['explanation'])

        except Exception as e:
            print(f"Error processing QA pair {idx + 1}: {e}")
            scores.append(None)
            expls.append("Error processing QA pair")
    return scores, expls


def save_scores(scores_dict, output_path):
    """
    Save the dictionary of scores to a JSON file next to the script.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scores_dict, f, indent=2)
    print(f"Scores saved to {output_path}")

def save_explanations(expl_dict, output_path):
    """
    Save the dictionary of explanation strings to a JSON file next to the script.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expl_dict, f, indent=2)
    print(f"Explanations saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA pairs using LLM.")
    parser.add_argument("--dir", help="Path to the input directory.")
    parser.add_argument("--file", help="Path to the file containing filter paths.")
    parser.add_argument("--model", help="Model ID to use for evaluation.", default="gemma3:27b")
    args = parser.parse_args()

    # Load filter paths from the file
    with open(args.file, 'r', encoding='utf-8') as f:
        filter_paths = set(json.load(f))

    # Find text and JSON files
    files_to_process = find_text_and_json_files(args.dir, filter_paths)
    if not files_to_process:
        print("No matching text and JSON file pairs found.")
        return

    scores_dict, expls_dict = {}, {}

    total = len(files_to_process.keys())
    cnt = 0
    output_path = os.path.join(os.path.dirname(__file__), "scores_summary12.json")

    for relative_json_path, (text_file, json_file) in files_to_process.items():
        cnt += 1
        print(f"\nProcessing {cnt}/{total}:\nText file: {text_file}\nJSON file: {json_file}")
        context = load_context(text_file)
        qa_pairs = load_qa_pairs(json_file)
        scores, explanations = evaluate_answers(args.model, context, qa_pairs)
        scores_dict[relative_json_path] = scores
        expls_dict[relative_json_path] = explanations

        if cnt % 10 == 0:
            save_scores(scores_dict, output_path)
            save_explanations(expls_dict, output_path.replace("scores_summary12.json", "explanations_summary12.json"))

    # Save scores to a single JSON file next to the script

    save_scores(scores_dict, output_path)
    save_explanations(expls_dict, output_path.replace("scores_summary12.json", "explanations_summary12.json"))

if __name__ == "__main__":
    main()

