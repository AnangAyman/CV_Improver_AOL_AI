import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2-medium'  # You can use 'gpt2-medium' for a larger model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

def improve_cv_text(input_text, max_length=500):
    # Tokenize the input text
    prompt = f"Improve this text for a CV: {input_text}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,          # Explicit attention mask
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id    # Set pad token explicitly
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Decode the generated text
    improved_text = generated_text[28:].strip()
    return improved_text