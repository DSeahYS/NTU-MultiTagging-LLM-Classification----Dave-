import os
from dotenv import load_dotenv
from reranker_core import create_client, classify_paper, SYSTEM_PROMPT

load_dotenv(dotenv_path="C:/VSCode Folder/NTU Visiting Researcher/1. Fine Tuning for AM/.env")
api_key = os.environ.get("openrouterkey", "")

print(f"API Key loaded: {api_key[:5]}...{api_key[-5:] if len(api_key)>10 else ''}")

client = create_client(api_key)
text = "This paper explores the use of extrusion-based bioprinting and GelMA to create functional cardiac tissue patches... " * 100

try:
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this paper text and assign tags:\n\n{text}"},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    print("RAW:", repr(raw))
    from reranker_core import _parse_llm_output
    tags = _parse_llm_output(raw)
    print("Tags:", tags)
except Exception as e:
    print("Error:", e)
except Exception as e:
    print("Error:", e)
