import transformers

import torch

from peft import LoraConfig

from icae.llama_icae_modeling import LlamaICAE, ModelArguments, DataArguments, TrainingArguments
from icae.llama_icae_learning import instruct_ft_tokenize_function

import json

device = "cuda"

memory_size = 128

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()

lora_config = LoraConfig(

    r=model_args.lora_r,

    lora_alpha=32,

    lora_dropout=model_args.lora_dropout,

    bias="none",

    task_type="CAUSAL_LM"

)

model = LlamaICAE(model_args, training_args, lora_config).to("cuda")

# icea model

state_dict = torch.load(
    "/path/to/your/checkpoint")

model.load_state_dict(state_dict)

from tqdm import tqdm

file_path = "/path/to/dev_v2.jsonl"

lines = None

with open(file_path, "r") as f:
    lines = f.readlines()

MEMORY_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

model.eval()

with torch.no_grad():
    with open("/path/to/output", "w") as f:

        for line in tqdm(lines):

            data = json.loads(line)

            text_output = model.tokenizer(data["input"], truncation=True,
                                          max_length=model.training_args.model_max_length, padding=False,
                                          return_attention_mask=False)

            text_output["input_ids"] += MEMORY_TOKENS

            input_ids = torch.tensor(text_output["input_ids"]).unsqueeze(0).to(device)

            batch_size = 1

            memory_mask = input_ids >= model.vocab_size

            autoencoder_input_embedding = model.icae.get_base_model().model.embed_tokens(input_ids)

            autoencoder_input_embedding[memopry_mask] = model.memory_token_embed(
                input_ids[memory_mask] - model.vocab_size).half()

            compress_outputs = model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                          enable_lora=True)

            compress_outputs = compress_outputs.hidden_states[-1]

            prompt_output = model.tokenizer(data['prompt'], add_special_tokens=False, padding=False)

            prompt_ids = [model.ft_token_id] + prompt_output['input_ids'] + [model.ft_token_id]

            prompt_answer_ids = torch.tensor([prompt_ids]).to(device)

            # get the last k hidden states

            memory_embedding = compress_outputs[memory_mask].view(batch_size, model.model_args.mem_size, -1)

            prompt_answer_embs = model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

            special_prompt = prompt_answer_ids >= model.vocab_size

            prompt_answer_embs[special_prompt] = model.memory_token_embed(
                prompt_answer_ids[special_prompt] - model.vocab_size).to(prompt_answer_embs)

            decoder_input_embeddings = torch.cat((memory_embedding, prompt_answer_embs), dim=1)

            output = decoder_input_embeddings.clone()

            generate_text = []

            past_key_values = None

            for i in range(512):

                out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)

                logit = out.logits[:, -1, :]

                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)

                if next_token_id == 1:
                    break

                if next_token_id.item() >= 32000:
                    break

                output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)

                generate_text.append(next_token_id.item())

            generated_text = model.tokenizer.decode(generate_text)

            output_ = {

                "text": None,

                "prompt": None,

                "answer": None,

                "output": None,

            }

            output_["text"] = data["input"]

            output_["prompt"] = data["prompt"]

            output_["answer"] = data["answer"]

            output_["output"] = generated_text

            f.write(json.dumps(output_) + "\n")
