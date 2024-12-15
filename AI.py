import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'google/flan-t5-base'  # You can use 'gpt2-medium' for a larger model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def improve_cv_text(input_text, max_length=1000):
    # Tokenize the input text
    prompt = f"Rewrite the following text to make it professional and concise for a CV: {input_text}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    print(1)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,          # Explicit attention mask
            max_length= max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id    # Set pad token explicitly
        )
        print(2)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(3)
    # Decode the generated text
    
    print(4)
    return generated_text
