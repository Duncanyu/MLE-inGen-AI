from openai import OpenAI
#CHANGE personal.personal TO config
from personal.personal import MAX_README_CHAR
from shared import query_val

query = query_val

def think(api_key, content, readme, model = "gpt-4"):
    client = OpenAI(api_key = api_key)
    
    newline = "\n"

    if len(readme) > MAX_README_CHAR:
        readme = readme[:MAX_README_CHAR] + "\n\n[TRUNCATED]"
    
    user_prompt = (
    "Summarize this repo:\n" +
    newline.join([f"{key}: {value}" for key, value in content.items()]) +
    f"\n\nREADME:\n{readme}"
    )
    
    response = client.chat.completions.create(
        model = model,
        messages=[
            {"role": "system", "content": f"You are a technical assistant. Be direct and professional. Respond ONLY with the summary without preamble or follow-up commentary. Focus on strenghts and weaknesses so they can be compared. Here is the original query: {query}"},
            {"role": "user", "content": user_prompt}
        ]
    )

    summary = response.choices[0].message.content
    
    return summary

def compare(api_key, repo_list, model="gpt-4"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    user_prompt = ("Compare the following GitHub repositories. Highlight their similarities, differences, and use cases. Consider factors such as: popularity, organization, officiality, etc... The final goal of the comparison is to recommend these 3 things: which one is best to use on a qucikstart standpoint, which one is best to use for more advanced consumers, which one to avoid all-together. If there is only one/two, then just give which one is best or whether the single repository is worth using or not. Output the comparison in clear, structured plain text. Use headings, bullet points, and clean formatting.\n\n")

    for i, repo in enumerate(repo_list):
        user_prompt += f"Repo {i + 1}:\n"
        for key, value in repo.items():
            if key != "summary":
                user_prompt += f"{key}: {value}\n"
        user_prompt += f"Summary: {repo.get('summary', 'No summary available')}\n\n"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a technical assistant. Be direct and professional. Respond ONLY with the comparison without preamble or follow-up commentary. Please only compare repos given to you, do not introduce any new repositories or alternatives. Here is the original query: {query}"},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()