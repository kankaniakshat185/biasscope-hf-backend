import os
from huggingface_hub import InferenceClient

hf_token = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
client = InferenceClient(model=model_id, token=hf_token)

total = 18
left_count = 15
center_count = 1
right_count = 2
pos_count = 3
neg_count = 12

context_str = """- Source: [indianexpress.com] | Headline: TMC grassroots machine under strain (Bias: RIGHT, Sentiment: negative)
- Source: [thehindu.com] | Headline: Mamata announces scheme (Bias: LEFT, Sentiment: positive)"""

system_prompt = (
    "You are an expert media analyst and political scientist. "
    "Your task is to write a highly professional, objective, and insightful 3-4 sentence narrative summary "
    "of the media's current coverage of a topic, based strictly on the provided article headlines, bias labels, and sentiments."
)

user_prompt = f"Media Analysis Data:\nTotal Articles: {total}\nBias Breakdown: {left_count} Left, {center_count} Center, {right_count} Right.\nSentiment: {pos_count} Positive, {neg_count} Negative.\n\nSample Articles:\n{context_str}\n\nPlease generate the executive summary narrative. \n\nCRITICAL: You MUST include inline citations for the sources using square brackets (Example: 'The narrative is largely negative [indianexpress.com], though some outlets highlight positive aspects [thehindu.com].'). Do NOT output any preambles like 'Here is a summary', just output the summary paragraph directly."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

print("Calling Llama-3...")
response = client.chat_completion(messages=messages, max_tokens=250, temperature=0.5)
print("RESPONSE:\n", response.choices[0].message.content.strip())
