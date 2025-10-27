"""
Evaluation script to generate predictions from trained LoRA models.
This creates the VALID_DATASET required for the analysis notebook.

IDENTICAL to utils/generate_dataset.py except:
1. Loads LoRA adapter on top of base model
2. Uses test dataset from trained model
3. Outputs to VALID_DATASET directory
"""
import transformers
import torch
from transformers import pipeline

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

def main(lora_path, base_model_path):
    """
    Generate predictions using the trained LoRA model.
    
    Args:
        lora_path: Path to trained LoRA adapter
        base_model_path: Path to base model
    """

    model_path = base_model_path

    N_SHOT = 4
    N_EXP = 10
    batch_size_greed = 128
    batch_size_sample = batch_size_greed // 16

    tokenizer = AutoTokenizer.from_pretrained(model_path,
        padding_side='left',
        )

    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        #attn_implementation="flash_attention_2",
        attn_implementation='eager',
        ).eval()
    
    # DIFFERENCE: Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    dataset_for_few = load_dataset("trivia_qa", "rc", 
                        trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        )

    def prompt_system_generator(dataset):
        end_text = []
        for question_text, answer_text in zip(*dataset):
            end_text.extend([{"role": "user", "content": "Question: " + question_text}])
            end_text.extend([{"role": "assistant","content": "Answer: " + answer_text["normalized_aliases"][0]}])
        return end_text

    def primer_system_user(examples):
        prompted_text = "Answer the following question."
        outputs = []
        
        # Precompute and reuse constant data
        questions = examples["question"]
        answers = examples["answer"]
        
        # Precompute primed_data
        primed_data = {
            "question": questions * N_EXP,
            "answer": answers * N_EXP,
        }
        
        # Preselect columns once
        train_dataset = dataset_for_few["train"].select_columns(["question", "answer"])

        # Precompute messages template for efficiency
        system_message = {"role": "system", "content": prompted_text}

        # Loop over add_iter once, calculating dataset_to_eval once per iteration
        for add_iter in range(N_EXP):
            start_index = N_SHOT * add_iter
            end_index = N_SHOT * (1 + add_iter)
            
            # Precompute dataset to evaluate
            dataset_to_eval = train_dataset[start_index:end_index].values()
            
            # Generate message list for each question in one go using list comprehension
            outputs.extend([
                pipe.tokenizer.apply_chat_template(
                    [system_message] + prompt_system_generator(dataset_to_eval) + [
                        {"role": "user", "content": f"Question: {question}"}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for question in questions
            ])

        primed_data["primed_question"] = outputs

        return primed_data

    # DIFFERENCE: Load test dataset from trained model instead of data_path
    train_dataset_dict = load_from_disk(os.path.join(lora_path, "dataset_to_train.dataset"))
    examples_unk2 = train_dataset_dict["test"]
    
    UNK_HN_dataset_90k = pd.DataFrame(examples_unk2)[['question','answer']]
    UNK_HN_dataset_90k['answer'] = UNK_HN_dataset_90k['answer'].apply( lambda a: a[0] if isinstance(a, list) else a)
    UNK_HN_dataset_90k_dataset = Dataset.from_pandas(UNK_HN_dataset_90k)

    primed_dataset = UNK_HN_dataset_90k_dataset.map(primer_system_user, batched=True,
                                        remove_columns=UNK_HN_dataset_90k_dataset.column_names,
                                        batch_size=1
                                        )
    
    def data():
        for i in tqdm(primed_dataset):
            yield i["primed_question"]

    new = []

    for out in pipe(
            data(),
            max_new_tokens=32,
            add_special_tokens = True,
            temperature=None, top_p=None, top_k=None,
            do_sample=False,
            batch_size=batch_size_greed,
            return_full_text=False,
            #truncation="only_first"
        ):
        new.append(out[0]['generated_text'])

    small_dataset = primed_dataset.add_column("greedy_ans", new)
    
    # DIFFERENCE: Actually generate sampled predictions (not just 'None')
    def data_sample():
        for i in tqdm(primed_dataset):
            yield i["primed_question"]
    
    sample_results = []
    for out in pipe(
            data_sample(),
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=5,
            batch_size=batch_size_sample,
            return_full_text=False,
        ):
        sample_results.append([o['generated_text'] for o in out])
    
    small_dataset = small_dataset.add_column("sample_ans", sample_results)

    def quasi_accuracy_triviaqa(samples):
        p_greed = []
        p_sample = []
        for answer, greedy_pred, sample_pred in zip(samples['answer'], samples['greedy_ans'], samples['sample_ans']):
            p_greed.append( any([greedy_pred.lower().find(i)+1 for i in answer['normalized_aliases'] ])  )
            p_sample.append( any([any([sample_i.lower().find(i)+1 
                                    for i in answer['normalized_aliases'] 
                                    ]) 
                                for sample_i in sample_pred
                                ]))
        samples['p_greed'] = p_greed
        samples['p_sample'] = p_sample
        return samples

    small_dataset = small_dataset.map(quasi_accuracy_triviaqa, batched=True,
                                        batch_size=128
                                        )
    def accuracy_check(list_of):
        if all(list_of['p_greed']):
            return 'HighlyKnown'
        elif any(list_of['p_greed']):
            return 'MaybeKnown'
        elif any(list_of['p_sample']):
            return 'WeaklyKnown'
        else:
            return 'Unknown'
        
    test_df = pd.DataFrame(small_dataset)
    test_df = test_df.groupby('question').agg(list).reset_index(drop=False)
    test_df['Category'] = test_df.apply(lambda a: accuracy_check(a), axis=1)

    from itertools import compress

    def filter_ans(layer_of):
        good_one = list(compress(layer_of['primed_question'], layer_of['p_greed'] ))[0]
        bad_one = list(compress(layer_of['primed_question'], [not i for i in layer_of['p_greed']] ))[0]
        return [bad_one, good_one]

    arr = test_df[test_df['Category'] == 'MaybeKnown'].apply(filter_ans ,axis=1)

    train_dataset = Dataset.from_pandas(test_df)

    # DIFFERENCE: Save to VALID_DATASET and include all splits
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": train_dataset,
        "valid": train_dataset,
        "full": train_dataset,
    })

    # DIFFERENCE: Save to VALID_DATASET directory
    output_path = os.path.join(lora_path, "VALID_DATASET")
    dataset_dict.save_to_disk(output_path)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Total examples: {len(train_dataset)}")
    print(f"\nCategory breakdown:")
    for cat in ['HighlyKnown', 'MaybeKnown', 'WeaklyKnown', 'Unknown']:
        count = (test_df['Category'] == cat).sum()
        print(f"  {cat}: {count} ({count/len(test_df)*100:.1f}%)")
    print(f"\nOutput saved to: {output_path}")
    print("\nYou can now run the analysis notebook!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained LoRA model and generate predictions for analysis."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA adapter directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Path to base model"
    )
    
    args = parser.parse_args()
    main(args.lora_path, args.base_model)
