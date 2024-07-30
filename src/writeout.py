"""
  PropertyExtractor -- LLM-based model to extract material property from unstructured dataset

  This program is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software Foundation
  version 3 of the License.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE.  See the GNU General Public License for more details.
  Email: cekuma1@gmail.com
  
""" 

import os
import json
from datetime import datetime


def write_default_inputs(cwd):
    """
    Writes a default 'extract.in' file in the specified directory if it does not exist.
    
    Parameters:
    cwd (str): The current working directory where the file should be written.
    """
    file_path = os.path.join(cwd, "extract.in")
    
    if not os.path.exists(file_path):
        extract_input = """
###############################################################################
### The input file to control the calculation details of PropertyExtract    ###
###############################################################################
# Type of LLM model: gemini/chatgpt 
model_type = gemini
# LLM model name: gemini-pro/gpt-4
model_name = gemini-pro
# Property to extract from texts
property = thickness
# Harmonized unit for the property to be extracted
property_unit = Angstrom
# temperature to max_output_tokens are LLM model parameters
temperature = 0.0
top_p = 0.95
max_output_tokens = 80
# You can supply additional keywords to be used in conjunction with the property: modify the file keywords.json
use_keywords = True
# You can add additional custom prompts: modify the file additionalprompt.txt
additional_prompts = additionalprompt.txt  
# Name of input file to be processed: csv/excel format     
inputfile_name = inputfile.csv
# Column name in the input file to be processed
column_name = Text
# Name of output file
outputfile_name = outputfile.csv
"""
        try:
            with open(file_path, "w") as file:
                file.write(extract_input.strip())  # Using .strip() to clean up leading/trailing whitespace
            print(f"'extract.in' created successfully in {cwd}.")
        except IOError as e:
            print(f"Failed to write to the file: {e}")
    else:
        print(f"'extract.in' already exists in {cwd}. No action was taken.")




def write_prep_keyword_prompts(cwd):
    """
    Writes a default prep_keyword.txt file in the specified directory if it does not exist.
    
    Parameters:
    cwd (str): The current working directory where the file should be written.
    """

    file_path = os.path.join(cwd, "prep_keyword.txt")

    if not os.path.exists(file_path):
        # Define the content to be written to the file
        pre_key = """["bandgap", "band gap"]\n"""

        try:
            with open(file_path, "w") as file:
                file.write(pre_key)
            print(f"'prep_keyword.txt' created successfully in {cwd}.")
        except IOError as e:
            print(f"Failed to write to the file: {e}")
    else:
        print(f"'prep_keyword.txt' already exists in {cwd}. No action was taken.")

        
        

def write_additionalprompts(cwd):
    """
    Writes a default additionalprompt.txt file in the specified directory if it does not exist.
    
    Parameters:
    cwd (str): The current working directory where the file should be written.
    """

    file_path = os.path.join(cwd, "additionalprompt.txt")
    

    if not os.path.exists(file_path):
        # Define the content to be written to the file
        additional_prompt_text = """         - Titanium dioxide films reported at 1 µm thickness: Report as "Material: TiO2, Thickness: 10".
        - Text mentions 2D-based h-AB monolayer with a thickness of 0.34 nm obtaied using AFM: Report as "Material: AB, Thickness: 3.4, Unit: nm, Method: AFM".
        - Text mentions the thickness of material "ABC3" is 60 Å  from our experimental data analysis: Report as "Material: ABC3, Thickness: 60, Unit: Å, Method: Experiment ".
"""

        try:
            with open(file_path, "w") as file:
                file.write(additional_prompt_text)
            print(f"'additionalprompt.txt' created successfully in {cwd}.")
        except IOError as e:
            print(f"Failed to write to the file: {e}")
    else:
        print(f"'additionalprompt.txt' already exists in {cwd}. No action was taken.")



def write_default_keywords(cwd):
    """
    Writes a default keywords.json file in the specified directory if it does not exist.
    
    Parameters:
    cwd (str): The current working directory where the file should be written.
    """
    # Define the path to the file
    file_path = os.path.join(cwd, "keywords.json")
    
    # Check if the file already exists
    if not os.path.exists(file_path):
        # Define the JSON data to be written to the file
        keywords_data = {
            "keywords": ["2D materials", "ultrathin materials", "van der Waals materials"],
            "keyword_weights": {
                "2D materials": "high",
                "ultrathin materials": "medium",
                "van der Waals materials": "high"
            },
            "keyword_synonyms": {
                "2D materials": ["two-dimensional materials"],
                "van der Waals materials": ["vdW materials"]
            }
        }
        
        # Write the JSON data to the file
        try:
            with open(file_path, "w") as file:
                json.dump(keywords_data, file, indent=4)
            print(f"'keywords.json' created successfully in {cwd}.")
        except IOError as e:
            print(f"Failed to write to the file: {e}")
    else:
        print(f"'keywords.json' already exists in {cwd}. No action was taken.")





def print_default_input_message_0():
    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                                ║")
    print("║                       ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥                       ║")
    print("║                ♥♥♥  Default extract.in input template generated. ♥♥♥           ║")
    print("║                 ♥♥ Modify and rerun thick2d -0 to generate other   ♥♥          ║")
    print("║                 ♥♥    important input files. Happy running :)    ♥♥            ║")
    print("║                       ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥                       ║")
    print("║                                   Exiting...                                   ║")
    print("║                                                                                ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝")





def print_default_input_message():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                          ║")
    print("║                   ♥♥    ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥   ♥♥                       ║")
    print("║                 ♥♥♥                               ♥♥♥                    ║")
    print("║                ♥ All default inputs written to files. ♥                  ║")
    print("║              ♥     Modify according LLM model/type      ♥                ║")
    print("║             ♥       Run code with propextract           ♥                ║")
    print("║              ♥          Happy running :)               ♥                 ║")
    print("║                 ♥♥♥                                ♥♥♥                   ║")
    print("║                       ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥                              ║")
    print("║                                Exiting...                                ║")
    print("║                                                                          ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")


max_width = len("|WARNING: This is an empirical approx; validity needs to be checked !! |")

def print_line(ec_file,content, padding=1, border_char="|", filler_char=" "):
    content_width = int(max_width) - (2 * int(padding)) - 2  # Subtract 2 for the border characters
    content = content[:content_width]  # Ensure content doesn't exceed the width
    line = border_char + filler_char*padding + content.ljust(content_width) + filler_char*padding + border_char
    #print(line)  # Print the line to the console
    if ec_file:
        ec_file.write(line + "\n")
    else:
        print(line)    

        
        


def print_banner(version,code_type,model_type,ec_file=None):
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = datetime.now().strftime('%Y-%m-%d')
    conclusion_msg = f"Calculations started at {current_time} on {current_date}"

    message = f"General languagge model simulations using \nPropertyExtract Version: {version}\n with {code_type} conversational LLM {model_type} model \nto perform simulations\n{conclusion_msg}"

    max_width = 80  # Define the maximum width for the banner

    print_line(ec_file,'❤' * (max_width - 2), padding=0, border_char='❤', filler_char='❤')
    for line in message.split('\n'):
        centered_line = line.center(max_width - 4)
        print_line(ec_file,centered_line, padding=1, border_char='❤')
    print_line(ec_file,'❤' * (max_width - 2), padding=0, border_char='❤', filler_char='❤')




def print_boxed_message(ec_file=None):
    header_footer = "+" + "-" * 78 + "+"
    spacer = "| " + " " * 76 + " |"

    # List of lines to be printed
    lines = [
        (" * CITATIONS *", True),
        ("If you have used PropertyExtractor in your research, PLEASE cite:", False),
        ("", False),  # Space after the above line
        ("PropertyExtractor: ", False),
        ("Dynamic in-context learning with conversational language", False),
        ("models for data extraction and materials property prediction ", False),
        ("C.E. Ekuma, ", False),
        ("XXX xxx, xxx, (2024)", False),
        ("", False),

        ("", False),  # Blank line for separation
        ("PropertyExtractor: ", False),
        ("A conversational large language model for extracting ", False),
        ("physical properties from scientific corpus, C.E. Ekuma,", False),
        ("www.github.com/gmp007/propertyextractor", False)
    ]

    def output_line(line):
        if ec_file:
            ec_file.write(line + "\n")
        else:
            print(line)

    output_line(header_footer)
    
    for line, underline in lines:
        centered_line = line.center(76)
        output_line("| " + centered_line + " |")
        
        if underline:
            underline_str = "-" * len(centered_line)
            output_line("| " + underline_str.center(76) + " |")

    # Print footer of the box
    output_line(header_footer)
