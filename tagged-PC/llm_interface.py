import os
import transformers
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub import login

"""
This class is for generating new tags via LLMs on huggingface. The default is Llama-3-8B-Instruct, feel free to try other models.
"""

def load_model_pipeline():

    os.environ['HF_HOME'] = '~/HuggingFace' #path you want to download model infos to
    os.environ['HF_HUB_CACHE'] = '~/HuggingFace' #path you want to download model to
    HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

    huggingface_api_key = "hf_xxxxxxxxxxxxxxxxxxxx" #XXX Put your huggingface API key here
    login(huggingface_api_key)

    # model specifications
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    filenames = [
        "config.json", "generation_config.json", "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "tokenizer_config.json"
    ]

    #pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    return pipeline


def text_generation(pipeline, prompt, deterministic = False):
    # Here Text Generation
    messages = [
        {"role": "system", "content": "You are an expert for assigning tags to Nodes used in causal inference"},
        {"role": "user", "content": prompt}, #change prompt here
    ]

    # tokenize prompt
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if deterministic:
        # generate deterministic output (at least as determinstic as possible, for some reason llama hast not implemented a seed paramete)
        outputs = pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.01, #low temperature
            top_k=1, #only take most likely option
            top_p=0, #only take from smallest possible subset
        )   
    else:
        # generate undeterminstic output
        outputs = pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    print(f"print full generated text: \n{outputs[0]["generated_text"][len(prompt):]}")
    return outputs[0]["generated_text"][len(prompt):]


def text_reduction_variable_tags(pipeline, generated_text, deterministic = False):
    # Answer Extracting part 1 tagstring
    messages = [
        {"role": "system", "content": "You are an oracle for causal inference that outputs only the requested information in a compact form, without any unnecessary phrases or introductions."},
        # Copy Answer here under the first line
        {"role": "user", "content": """ Please Shorten your Answer to each Variable followed by all fitting tags. So that you get the form: 
        <Variable1> : <Tag1>, <Tag3>... 
        <Variable2> : <Tag3>...
        (substitute the words in the <> with the fitting words)
        Your Answer was the following:
        """ + generated_text + """ 
        remember that every Variable can have **multiple fitting tags**, refrain from using unnecessary words - your message should **start directly with** - and **only contain all the variables and their respective tags** in the form above. **Do not stop generating before listing ALL variables**"""},
        #change prompt here
    ]

    # tokenize prompt
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if deterministic:
        # generate deterministic output (at least as determinstic as possible, for some reason llama hast not implemented a seed paramete)
        outputs = pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.01, #low temperature
            top_k=1, #only take most likely option
            top_p=0, #only take from smallest possible subset
        )   
    else:
        # generate undeterminstic output
        outputs = pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    print(f"generated tags: \n{outputs[0]["generated_text"][len(prompt):]}")
    return outputs[0]["generated_text"][len(prompt):]


def text_reduction_taglist(pipeline, generated_text, deterministic = False):
    # Answer Extracting part 2 get taglist
    messages = [
        {"role": "system", "content": "You are an oracle for causal inference that outputs only the requested information in a compact form, without any unnecessary phrases or introductions."},
        # Copy Answer here under the first line
        {"role": "user", "content": """ Please Shorten your Answer to just a list of all tags, only seperated by commata, so that you get the form: 
        <Tag1>, <Tag2>, <Tag3>, .... <Tagn>
        (substitute the words in the <> with the fitting words)
        Your Answer was the following:
        """ + generated_text + """ 
       refrain from using unnecessary words - your message should **start directly with** - and **only contain all the tags** in the form above,"""}, 
        #change prompt here
    ]

    # tokenize prompt
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if deterministic:
        # generate deterministic output (at least as determinstic as possible, for some reason llama hast not implemented a seed paramete)
        outputs = pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.01, #low temperature
            top_k=1, #only take most likely option
            top_p=0, #only take from smallest possible subset
        )   
    else:
        # generate undeterminstic output
        outputs = pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    print(f"generated taglist: \n{outputs[0]["generated_text"][len(prompt):]}")
    return outputs[0]["generated_text"][len(prompt):]
