import json
import pandas as pd
import re
import os

data = []

pattern = re.compile(r"English\s*:\s*(.*?)\nDarija\s*:\s*(.*)")

with open('darija_results_not_extracted16.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
            
        parsed_json = json.loads(line)
        
        try:
            text_blob = parsed_json["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except KeyError:
            print("Skipping a line: Could not find the expected text path in the JSON structure.")
            continue
        
        blocks = text_blob.split("===")
        
        for block in blocks:
            match = pattern.search(block)
            if match:
                english_sentence = match.group(1).strip()
                darija_sentence = match.group(2).strip()
                
                data.append({
                    "english": english_sentence,
                    "darija": darija_sentence
                })

df = pd.DataFrame(data)

csv_filename = 'darija_translations.csv'
file_exists = os.path.exists(csv_filename)

df.to_csv(csv_filename, mode='a', index=False, encoding='utf-8', header=not file_exists)

if file_exists:
    print(f"Successfully appended {len(df)} translation pairs to '{csv_filename}'.")
else:
    print(f"Successfully extracted {len(df)} translation pairs and created '{csv_filename}'.")