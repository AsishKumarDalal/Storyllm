from ..tokenizer.encode_decode import *
def generate_and_print_sample__(model, tokenizer, device, start_context):
    """Generate and print a sample text"""
    model.eval()
    context_size = 128
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = model.generate(
            idx=encoded,
            max_new_tokens=300,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()