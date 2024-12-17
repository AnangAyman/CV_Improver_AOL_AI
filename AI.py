import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from heuristics import heuristic_score


model_name = 'bigscience/bloom-560m'  # You can use 'gpt2-medium' for a larger model
tokenizer =  BloomTokenizerFast.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

def improve_cv_text(input_text = "I am a critical thinker, i love problem solving and a very logical person which makes me ultimately thinks objectively", max_length=1000):
    # Tokenize the input text
    print(f"Score before: {heuristic_score(input_text)}")
    
    prompt = f"Rewrite the following CV text to make it more professional and error-free: {input_text}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,          # Explicit attention mask
            max_length=max_length,
            num_beams = 15,
            do_sample=True,
            temperature = 0.3,
            num_return_sequences=1,
            early_stopping = False,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id    # Set pad token explicitly
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Decode the generated text
    improved_text = generated_text[0:].strip()

    final_result = improved_text
    print(f"Score after: {heuristic_score(final_result)}")

    print(f"{improved_text}")
    return final_result