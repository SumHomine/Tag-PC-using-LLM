import os
import transformers
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub import login

from tag_pc_utils import get_taglist_from_llm_output

# This File differes from the OG one in Type PC, by also returning tag_list in the used function
"""
This class is for generating new tags via LLMs on huggingface. The default is Llama-3-8B-Instruct, feel free to try other models.
"""


def run_llm_generic_prompt_typed(node_names : list, determinstic = True):
    """
    use this method to get tags using a LLM 
    :param prompt: prompt that is fed to the LLM, will independently then reduce it and put in the pipeline
    :returns types: String with each line being the node name followed by colon and the type in the form:
        Cloudy : Weather
        Sprinkler : Watering
        Rain : Weather
        Wet_Grass : Plant_Con  
    :returns node_names: unchanged node_name input for consistency to other run_llm methods
    """

    #run LLM
    pipeline = load_model_pipeline()
    prompt = f"""We have found a causal system consisting of {len(node_names)} variables. Your job is now to assign those factors with each one tag. Please think of a handful of recurring characteristika to use as tags and then iteratively assign to each variable the fitting tag, so that **each variable** has **one and only one tag**. The variables are the following {node_names}"""
    print(prompt)
    out = text_generation(pipeline, prompt, determinstic)
    types_string = text_reduction_variable_types(pipeline, out, determinstic)
    typelist_string = text_reduction_typelist(pipeline, out, determinstic)
    #process LLM Output:
    type_list = get_taglist_from_llm_output(typelist_string) #turn taglist to actual list
 #   tags = recover_tag_string_onevsall_using_taglist(tags_string=tags_string, tag_list=tag_list, node_names=node_names)
    
    return types_string, type_list, node_names


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


def text_reduction_variable_types(pipeline, generated_text, deterministic = False):
    # Answer Extracting part 1 tagstring
    messages = [
        {"role": "system", "content": "You are an oracle for causal inference that outputs only the requested information in a compact form, without any unnecessary phrases or introductions."},
        # Copy Answer here under the first line
        {"role": "user", "content": """ Please Shorten your Answer to each Variable followed by its fitting tag. So that you get the form: 
        <Variable1> : <Type1> 
        <Variable2> : <Type2>
        <Variable2> : <Type1>
        (substitute the words in the <> with the fitting words)
        Your Answer was the following:
        """ + generated_text + """ 
        remember that every Variable must have one and only one tag, refrain from using unnecessary words - your message should **start directly with** - and **only contain the variables and tags** in the form above."""},
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


def text_reduction_typelist(pipeline, generated_text, deterministic = False):
    # Answer Extracting part 2 get taglist
    messages = [
        {"role": "system", "content": "You are an oracle for causal inference that outputs only the requested information in a compact form, without any unnecessary phrases or introductions."},
        # Copy Answer here under the first line
        {"role": "user", "content": """ Please Shorten your Answer to just a list of all tags, only seperated by commata, so that you get the form: 
        <Type1>, <Type2>, <Type3>, .... <Typen>
        (substitute the words in the <> with the fitting words)
        Your Answer was the following:
        """ + generated_text + """ 
       refrain from using unnecessary words - your message should **start directly with** - and **only contain the tags** in the form above,"""}, 
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