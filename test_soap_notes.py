# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog
import os

model = "LLAMA2_7B"

current_directory = os.path.dirname(os.path.abspath(__file__))
transcript_folder = "/home/ubuntu/daily/llama-rs-hacks/transcript_10-04_10"
soap_folder = transcript_folder.replace("transcript", "soap")
soap_folder = model + "_" + soap_folder

transcript_path = os.path.join(current_directory, transcript_folder)
soap_path = os.path.join(current_directory, soap_folder)

# Create soap notes folder if needed
os.makedirs(soap_folder, exist_ok=True)


file_list = os.listdir(transcript_path)
transcripts = [file for file in file_list if file.startswith("tr_") and file.endswith(".txt")]

transcript_path = "/home/ubuntu/daily/llama-rs-hacks/transcript_10-04_10"
txt_file = "tr_0_688w.txt"

system_context = """
You are a nurse practitioner with over 20 years of experience
writing clear, concise, and accurate SOAP notes for doctors. 
Format the SOAP note into ONLY these four sections: 
Subjective, Objective, Assessment, Plan. Only include facts 
about the patient's symptoms exactly as they are stated in a 
given transcript 'text' between doctor and patient in the 
Objective section.
"""


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    for i, txt_file in enumerate(transcripts):
        print(f"[{i+1}] : Reading {txt_file}:")
        transcript = None
        with open(os.path.join(transcript_path, txt_file), "r") as file:
            transcript = file.read()

        if not transcript:
            print(f"Skipping {txt_file} ...")
            continue

        prompt = f"Transcript: `{transcript}`"
        print("[{i+1}] : done. prompt ready....")

        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": system_context},
                {"role": "user", "content": prompt},
            ],
        ]

        print(f"[{i+1}] : sending query to {model}")
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # for dialog, result in zip(dialogs, results):
        #     for msg in dialog:
        #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        #     print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        #     print("\n==================================\n")

        content = results[0]["generation"]["content"]
        soap_file = txt_file.replace("tr_", "sp_")
        soap_file_path = os.path.join(soap_folder, soap_file)

        print(f"[{i+1}] : saving response")
        # Open the file in write mode ('w')
        with open(soap_file_path, "w") as file:
            # Write the string to the file
            file.write(content)

        break


if __name__ == "__main__":
    fire.Fire(main)
