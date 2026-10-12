import json

THEMES_FILE = "themes.txt"
PROMPT_TEMPLATE_FILE = "prompt.txt"
OUTPUT_FILE = "prompts_to_send.jsonl"

def generate_batch_file():
    with open(PROMPT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
        template = f.read()
    
    with open(THEMES_FILE, "r", encoding="utf-8") as f:
        themes = [line.strip() for line in f if line.strip()]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i, theme in enumerate(themes):
            full_prompt = template.replace("{}", theme)
            
            batch_request = {
                "key": f"request_{i}",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": full_prompt}
                            ]
                        } ],
                    "generation_config": {"temperature": 0.7, "top_p": 0.95}
                    
                }
            }
            
            out_f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(themes)} requests in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_batch_file()