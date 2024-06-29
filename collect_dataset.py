import JailMine
import torch
import torch.nn as nn
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument('--dataset', type=str, default="advBench")
FLAGS = parser.parse_args()


def main(args):
    # load the HF token from "../keys.txt", the line where starts with "Huggingface: "
    keys_file = open("../keys.txt", "r")
    for line in keys_file:
        if line.startswith("Huggingface: "):
            os.environ['HF_TOKEN'] = line.strip().split(": ")[1]
            break

    # Load the models, and the sorting file path
    MODEL_NAME='mistral-7b'                           # For Hooked Transformer to edit the logits
    MODEL_PATH = '../huggingface_cache/models--mistralai--Mistral-7B-v0.1'          # Victim Model, used to be mistralai/Mistral-7B-v0.1
    REPHRASE_PATH = '../huggingface_cache/models--meta-llama--Llama-2-7b-chat-hf'   # Attacker Model
    JUDGE_PATH = 'models--meta-llama--Meta-Llama-Guard-2-8B'                        # Judge Model
    SORTING_PATH = 'sorting.pth'
    EMBED_PATH='sentence-transformers/gtr-t5-xl'

    # Load the question set
    question_path = f'./datasets/{args.dataset}.csv'
    questions_set = pd.read_csv(question_path)['goal'].tolist()

    # Initialize the miner
    n_devices = torch.cuda.device_count()
    miner = JailMine.JailMine(
        model_name = MODEL_NAME,
        target_model_path = MODEL_PATH,
        rephrase_model_path = REPHRASE_PATH,
        sorting_model_path = SORTING_PATH,
        embedding_model_path = EMBED_PATH,
        judge_model_path = JUDGE_PATH,
        n_devices = n_devices,
    )

    # Run
    print(f"Starts to run. There are {len(questions_set)} questions in total.")
    miner.run(questions = questions_set, m=5, N=2000, output_path="./jailMine_advBench.csv")


if __name__=="__main__":
    main(FLAGS)