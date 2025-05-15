import json

def process_file(file_path, output_file):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for i in range(0, len(lines), 4):
        # Extract sentence and remove the number and quotes
        sentence_line = lines[i].strip().split('\t')
        sentence = sentence_line[1].strip('"') if len(sentence_line) > 1 else ""

        # Extract relation
        relation = lines[i+1].strip() if (i + 1) < len(lines) else "Other"

        data.append({"sentence": sentence, "relation": relation})

    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

process_file('./raw/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', './combined/train_full.json')
process_file('./raw/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', './combined/test_full.json')
