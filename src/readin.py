"""
  PropertyExtract -- LLM-based model to extract material property from unstructured dataset

  This program is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software Foundation
  version 3 of the License.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE.  See the GNU General Public License for more details.
  Email: cekuma1@gmail.com
  
""" 
import os
import sys
import spacy
from writeout import write_default_inputs,write_additionalprompts,write_default_keywords,print_default_input_message,write_prep_keyword_prompts
 
def get_version():
    try:
        from importlib.metadata import version  # Python 3.8+
        return version("PropertyExtract")
    except ImportError:
        try:
            from importlib_metadata import version  # Python <3.8
            return version("PropertyExtract")
        except ImportError:
            import pkg_resources
            return pkg_resources.get_distribution("PropertyExtract").version
            


def load_additional_prompts(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding='utf-8') as file:
                return file.read().strip()
        except IOError as e:
            print(f"An error occurred while reading the file: {e}")
    else:
        print(f"You have not provided additional prompt file, '{filename}'. Make sure that's what you want!")
    return ""
    
                
def read_input():
    """
    Read the application configuration from an input file and environment variables.
    
    Returns:
    - Dictionary containing configuration settings.
    """
    model_type = os.getenv('MODEL_TYPE', 'gemini').lower()
    api_key_env_var = 'GEMINI_PRO_API_KEY' if model_type == 'gemini' else 'OPENAI_API_KEY'
    api_key = os.getenv(api_key_env_var)
    
    if not api_key:
        print(f"API key for {model_type} is not set.")
        print(f"Please set it as: export {api_key_env_var}='your_actual_api_key_here'")
        sys.exit(1)

    cwd = os.getcwd()
    ppt_extract_exist = os.path.exists(os.path.join(cwd, "extract.in"))
    run_mode_flag = (len(sys.argv) > 1 and sys.argv[1] == "-0")
    if run_mode_flag and not ppt_extract_exist:
        write_default_inputs(cwd)
        write_additionalprompts(cwd)
        write_default_keywords(cwd)
        write_prep_keyword_prompts(cwd)
        print_default_input_message()
        sys.exit(0)

        
    """
    Read the stress component options from the 'extract.in' file.
    ...
    rotation = on/off
    ...
    Returns:
    - Dictionary of the component settings.
    """
    options = {
        'api_key': api_key,
        'model_type': 'gemini',
        'model_name': 'gemini-pro',
        'property_name': 'thickness',
        'property_unit': 'Angstrom',
        'temperature': 0.0,
        'top_p': 0.95,
        'max_output_tokens': 80,
        'use_keywords': False,
        'additional_prompts': None,       
        'inputfile_name': None,
        'outputfile_name': None,
        'column_name': 'Text',
    }

    try:
        with open("extract.in", "r") as f:
            lines = f.readlines()
           # previous_line = None
            for line in lines:
                line = line.strip()
                if line.startswith("#") or not line:
                    #previous_line = line.strip()
                    continue
                key, value = line.strip().split('=', 1)
                key = key.strip()
                value = value.strip()

                if key in ["inputfile_name", "outputfile_name", "additional_prompts", "column_name", "property_name"]:
                    options[key] = value
                elif key in ["model_type","model_name", "property_unit"]:
                    options[key] = value.lower()
                elif key in ["use_ml_model", "use_keywords"]:
                    options[key] = value.lower() in ['true', 'yes', '1','on']
                elif key in options:
                    if key in ['temperature','top_p','max_output_tokens']:
                        options[key] = float(value)
                    else:
                        options[key] = value.lower() == 'on'
                else:
                    #options['custom_options'][key] = value
                    options.setdefault('custom_options', {})[key] = value

        #if options.get('job_submit_command'):
        #    os.environ["ASE_VASP_COMMAND"] = options['job_submit_command']

    except FileNotFoundError:
        print("'extract.in' file not found. Using default settings.")
        
    #model_type = options.get("model_type")
    run_mode_flag = (len(sys.argv) > 1 and sys.argv[1] == "-0") #and 'dimensional' in options
    if run_mode_flag and ppt_extract_exist:
        write_default_inputs(cwd)
        print_default_input_message()
        sys.exit(0)
        
    return options


def ensure_spacy_model(model_name="en_core_web_sm"):
    """Ensure that the spaCy model is downloaded and available."""
    try:
        # Try to load the model to see if it's already installed
        spacy.load(model_name)
        print(f"Model {model_name} is already installed.")
    except OSError:
        # If the model isn't installed, download it
        print(f"Downloading the spaCy model {model_name}...")
        spacy.cli.download(model_name)
        
        
def configure_api(model_type):
    if model_type.lower() == 'gemini':
        import google.generativeai as genai
        gemini_pro_api_key = os.getenv('GEMINI_PRO_API_KEY')
        if not gemini_pro_api_key:
            print("GEMINI_PRO_API_KEY environment variable not set.")
            sys.exit(1)
        try:
            genai.configure(api_key=gemini_pro_api_key)
            print("Model Configured to use Gemini Pro with provided API key.")
        except Exception as e:
            print(f"Failed to configure Gemini Pro: {e}")
            sys.exit(1)
    elif model_type.lower() == 'chatgpt':
        gpt4_api_key = os.getenv('OPENAI_API_KEY')
        if not gpt4_api_key:
            print("OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = gpt4_api_key
        print("Configured for OpenAI with provided API key.")
    else:
        print("Invalid model type specified. Please choose 'gemini' or 'chatgpt'.")
        sys.exit(1)





def configure_apiold(model_type, api_key):
    if model_type.lower() == 'gemini':
        import google.generativeai as genai
        try:
            genai.configure(api_key=api_key)
            print("Model Configured to use Gemini Pro with provided API key.")
        except Exception as e:
            print(f"Failed to configure Gemini Pro: {e}")
            sys.exit(1)
    elif model_type.lower() == 'chatgpt':
        os.environ["OPENAI_API_KEY"] = api_key
        print("Configured for ChatGPT with provided API key.")
    else:
        print("Invalid model type specified. Please choose 'gemini' or 'chatgpt'.")
        sys.exit(1)
