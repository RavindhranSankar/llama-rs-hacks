import os
import datetime
import json

current_directory = os.path.dirname(os.path.abspath(__file__))
transcript_path = os.path.join(current_directory, "transcript_10-04_10")
soap_path = os.path.join(current_directory, "gpt-4_soap_10-04_10")

# prompt_template
prompt_template = """### Transcript:
{transcript}

### Soap Notes: """

max_num = 999

file_list = os.listdir(transcript_path)
transcripts = [file for file in file_list if file.startswith("tr_") and file.endswith(".txt")]
tr2sp = lambda t: t.replace("tr", "sp")

timestamp = datetime.datetime.now().strftime("%m_%d_%H")
jsonl_file = f"training_data_{timestamp}.jsonl"
dataset = []

# Loop thru all transcripts
for i, tfile in enumerate(transcripts[:max_num]):
    spfile = tr2sp(tfile)
    soap_file_path = os.path.join(soap_path, spfile)

    # Look for a corresponding gpt4 soap note
    if not os.path.exists(soap_file_path):
        continue

    print(f"[{i+1}] : reading {tfile} ...")
    transcript = None
    with open(os.path.join(transcript_path, tfile), "r") as file:
        transcript = file.read()
    if not transcript:
        continue

    print(f"[{i+1}] : looking for {spfile} ...")
    soap = None
    with open(os.path.join(soap_path, spfile), "r") as file:
        soap = file.read()
    if not soap:
        continue

    print(f"[{i+1}] : found!")

    # Insert transcript and soap note in template
    text = prompt_template.format(transcript=transcript)
    json_str = json.dumps({"user": text, "output": soap})

    print(f"[{i+1}] : updating {jsonl_file}!")
    with open(jsonl_file, "a") as file:
        file.write(json_str + "\n")

    print(f"[{i+1}] : done -------------")
