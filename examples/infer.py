"""Minimal inference example for a published, Transformers-compatible full model."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Ask a question about supplied paper text.")
    parser.add_argument("--model", required=True, help="Full model ID or local model directory")
    parser.add_argument("--context", required=True, help="Paper text supplied to the model")
    parser.add_argument("--question", required=True, help="Question about the paper")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    messages = [{"role": "user", "content": f"论文内容：\n{args.context}\n\n问题：{args.question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
