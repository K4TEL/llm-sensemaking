import csv
import json
import os

import ollama
# Assuming ollama client is installed and configured
# from ollama import Client # Corrected import if using official client
# from ollama import Ollama # Using the provided import style

# --- Configuration ---
# File paths
input_file = "./qa_data_55578.csv" # <--- Change to your input CSV file path
output_file = "./dataset.csv" # <--- Desired output CSV file path
# Ollama model
ollama_model_name = "llama3.3:latest" # <--- Replace with your actual Ollama model name

# --- Ollama Interaction ---
def get_ollama_response(question: str, answer: str, context: str, model_name: str) -> dict:
    """
    Use Ollama client to get score and explanation for an answer given a question and context.

    Args:
        question: The question asked.
        answer: The answer provided (potentially incorrect/neighboring).
        context: The generated context (potentially "silly").
        model_name: The name of the Ollama model to use.

    Returns:
        A dictionary containing the 'score' (0-100) and 'explanation'.
    """
    # Initialize the client within the function or globally if preferred and safe
    # client = Client(host='http://localhost:11434') # Example for official client
    # client = ollama.Client() # Using the provided import style
    
    answer = answer[2:-2] if answer.startswith("[") else answer

    # Updated prompt including context
    prompt = (
        f"Given the following context:\n{context}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Please evaluate the quality of the *Answer* specifically in response to the *Question*, considering the *Context* provided. "
        f"Provide your response as a JSON object with two keys and ',' delimiter: 'score' (an integer between 0 and 100, where 100 is best) "
        f"and 'explanation' (a string briefly justifying the score, noting if the answer is irrelevant to the question or context). No extra notes outside JSON in the output."
        # Example for forcing JSON output if model supports it:
        # f" Format your entire response as a single JSON object."
    )
    
    # print(prompt, "---\n")

    try:
        # response = client.generate(model=model_name, prompt=prompt) # Official client example
        # response_text = response['response'] # Official client example

        # Using the provided generate structure
        response = ollama.generate(model=model_name, prompt=prompt)
        response_text = response['response'] # Assuming 'response' key holds the generated text

        print(response_text)
        # Attempt to parse the response as JSON
        cleaned_response_text = response_text[next(idx for idx, c in enumerate(response_text) if c in "{["):]
        result = json.loads(cleaned_response_text)
        score = int(result.get("score", 0))
        explanation = result.get("explanation", response_text)
        # Clamp score to 0-100 range
        score = max(0, min(100, score))

    except Exception as e:
        print(f"Error during Ollama API call: {e}")
        score = -1 # Indicate error
        explanation = f"Error interacting with Ollama: {e}"

    return {"score": score, "explanation": explanation}

# --- Data Manipulation Functions ---
def get_neighboring_answer(current_index: int, all_data: list) -> str:
    """
    Gets the answer from the next row, wrapping around to the first if necessary.

    Args:
        current_index: The index of the current row.
        all_data: The list containing all data rows (as dictionaries).

    Returns:
        The answer string from the neighboring row. Returns empty string if data is empty.
    """
    num_rows = len(all_data)
    if num_rows == 0:
        return ""
    if num_rows == 1:
        # No neighbor, return its own answer (or empty string)
        return all_data[0].get("answer", "")

    neighbor_index = (current_index + 1) % num_rows
    return all_data[neighbor_index].get("answer", "") # Use .get for safety

def generate_silly_context(current_index: int, all_data: list) -> str:
    """
    Generates context by concatenating passages from the previous, current, and next rows.

    Args:
        current_index: The index of the current row.
        all_data: The list containing all data rows (as dictionaries).

    Returns:
        A single string containing the concatenated passages. Returns empty string if data is empty.
    """
    num_rows = len(all_data)
    if num_rows == 0:
        return ""

    # Get current passage
    current_passage = all_data[current_index].get("passages", "")

    # Get previous passage (handle boundary case: first row)
    prev_index = (current_index - 1 + num_rows) % num_rows # Wraps around correctly
    prev_passage = all_data[prev_index].get("passages", "") if num_rows > 1 else ""

    # Get next passage (handle boundary case: last row)
    next_index = (current_index + 1) % num_rows # Wraps around correctly
    next_passage = all_data[next_index].get("passages", "") if num_rows > 1 else ""

    # Concatenate - handle cases where passages might be missing/empty
    # Add separators for clarity
    context_parts = []
    if prev_passage and current_index != prev_index: # Avoid duplicating if only one row
        context_parts.append(f"{prev_passage}")
    if current_passage:
        context_parts.append(f"{current_passage}")
    if next_passage and current_index != next_index: # Avoid duplicating if only one row
         context_parts.append(f"{next_passage}")

    return "\n".join(context_parts)


# --- Main Processing Function ---
def process_data_with_ollama(input_path: str, output_path: str, model_name: str):
    """
    Reads CSV data, generates silly context and uses neighboring answers,
    gets scores/explanations from Ollama, and writes to a new CSV.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path to save the output CSV file.
        model_name: Name of the Ollama model.
    """
    all_data = []
    try:
        with open(input_path, 'r', encoding='utf-7', newline='') as infile:
            reader = csv.DictReader(infile)
            # Check required columns exist (adjust if needed)
            required_cols = ['query_id', 'question', 'answer', 'passages']
            if not all(col in reader.fieldnames for col in required_cols):
                missing = [col for col in required_cols if col not in reader.fieldnames]
                raise ValueError(f"Input CSV is missing required columns: {missing}")
            for row in reader:
                all_data.append(row)
        print(f"Read {len(all_data)} rows from {input_path}")
        if not all_data:
            print("Input file is empty. No processing done.")
            return

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file {input_path}: {e}")
        return

    # Prepare output CSV
    output_fieldnames = ['query_id', 'question', 'answer', 'context', 'score', 'explanation']
    processed_count = 0

    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()

            for i, row in enumerate(all_data):
                print(f"Processing row {i+1}/{len(all_data)}...")
                query_id = row.get('query_id', '')
                question = row.get('question', '')
                original_answer = row.get('answer', '') # Keep if needed, but not used for scoring
                # passages = row.get('passages', '') # Keep if needed

                # 1. Get the neighboring answer
                neighbor_answer = get_neighboring_answer(i, all_data)

                # 2. Generate the silly context
                silly_context = generate_silly_context(i, all_data)

                print(f"Calling llama for row {i+1}/{len(all_data)}...")
                # 3. Get score and explanation from Ollama using the question, *neighboring* answer, and *silly* context
                if question and neighbor_answer: # Only call Ollama if we have key inputs
                    score_explanation = get_ollama_response(question, neighbor_answer, silly_context, model_name)
                else:
                     print(f"Skipping Ollama call for row {i+1} due to missing question or neighbor answer.")
                     score_explanation = {"score": 0, "explanation": "Skipped Ollama call due to missing input."}


                # 4. Prepare data for output CSV
                output_row = {
                    'query_id': query_id,
                    'question': question,
                    'answer': neighbor_answer, # Write the neighbor answer that was scored
                    'context': silly_context, # Write the context used for scoring
                    'score': score_explanation['score'],
                    'explanation': score_explanation['explanation']
                }

                # 5. Write the processed row to the output CSV
                writer.writerow(output_row)

                score_explanation = get_ollama_response(question, original_answer, silly_context, model_name)

                output_row = {
                    'query_id': query_id,
                    'question': question,
                    'answer': original_answer,  # Write the neighbor answer that was scored
                    'context': silly_context,  # Write the context used for scoring
                    'score': score_explanation['score'],
                    'explanation': score_explanation['explanation']
                }

                # 5. Write the processed row to the output CSV
                writer.writerow(output_row)
                print(output_row)

                processed_count += 1

        print(f"Successfully processed {processed_count} rows.")
        print(f"Expanded data saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during processing or writing the output CSV: {e}")
        # Consider adding more specific error handling if needed

# --- Execution ---
if __name__ == "__main__":
    # Make sure the input file exists before running
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found. Please check the path.")
    else:
        process_data_with_ollama(input_file, output_file, ollama_model_name)
