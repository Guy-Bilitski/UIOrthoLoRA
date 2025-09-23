import json
import random

def augment_scores_in_jsonl_fast(input_path, output_path, advanced_models):
    """
    Efficiently augment a JSONL file with synthetic scores and empty training flags.
    
    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to save the augmented JSONL file.
        advanced_models (list): List of advanced model names to add scores for.
    """
    score_choices = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    model_fields = [(f"{model}_score", f"{model}_trained") for model in advanced_models]

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            data["base_model_score"] = random.choice(score_choices) # add base model score on this row
            for score_key, trained_key in model_fields:
                data[score_key] = random.choice(score_choices) # add advanced model score on this row
                data[trained_key] = "" # add T/F if this id row was used for training
            outfile.write(json.dumps(data) + "\n")

# add example how to use the function:     
#augment_scores_in_jsonl_fast(
#   input_path="triviaqa_q&a_first10.jsonl", # path to the input file
#    output_path="triviaqa_augmented.jsonl", # path to the output file
#    advanced_models=["adv_model_1", "adv_model_2"] # list of advanced models to add scores for
#)
