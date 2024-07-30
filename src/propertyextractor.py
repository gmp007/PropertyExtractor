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


#@title Multishot and Dynamical LLM
#Multishot dynamical approach - Gemini
import pandas as pd
import re
import os
import csv
import time
import json
import traceback
from datetime import datetime
import logging
import random
from textblob import TextBlob
import spacy
#from googleapiclient.errors import HttpError
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import google.generativeai as genai
from google.generativeai import GenerationConfig
import openai
from openai import OpenAI
#from openai.error import OpenAIError
#from openai.error import RateLimitError
from chardet import detect
from urllib3.exceptions import MaxRetryError, ReadTimeoutError
from mendeleev import element as get_element
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
import backoff #Solve "HTTP/1.1 429 Too Many Requests"


class PropertyExtractor:
    def __init__(self, property_name, property_unit, model_name='gemini-pro', model_type='gemini', temperature=0.0, top_p =0.95, max_output_tokens=80, additional_prompts=None,keyword_filepath=None,prep_keyword_path=None):
        self.property = property_name.lower() #self._create_property_regex(property_name)  
        self.unit_conversion = property_unit
        self.additional_prompts = additional_prompts
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.recent_analyses = []  # To store recent analyses for a multi-shot approach
        self.keyword_filepath = keyword_filepath
        self.keywords = self.load_keywords() or {}
        self.keywords_str = self._build_keywords()
        self.prep_keyword_path = prep_keyword_path
        

        if self.prep_keyword_path:
            print("Post processing keywords read successfully")
                    
        
        if self.load_keywords():
            print("Keywords read successfully!")
            
        if self.additional_prompts:
            print("Additional prompts read successfully!")
            

        
        
        if self.model_type == 'gemini':
            self.model = genai.GenerativeModel(self.model_name)
            self.generation_config = GenerationConfig(
            stop_sequences= None, # ["Keywords:", "Introduction", "Conclusion"],  # Adjust as necessary
            temperature=self.temperature,  # Balanced for precision and flexibility
            top_p=self.top_p,  # Broad yet focused selection of tokens
            top_k=50,  # Considers a wide range of possible tokens
            candidate_count=1,  # Single, well-considered response (increase if comparison needed)
            max_output_tokens=self.max_output_tokens  # Enough for detailed responses and explanations
            )
        elif self.model_type == 'chatgpt':
            self.model = OpenAI()
        else:
            raise ValueError("Unsupported model type provided. Use 'gemini' or 'gpt-4'.")


        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'POST'])
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.elements = {}
        #{
        #'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        #'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        #'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        #'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
        #'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        #'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
        #'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        #'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        #'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, #'Th': 90,
        #'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
        #'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
        #'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        #}



    def load_keywords(self):
        """Load keywords from a JSON file, checking if the file is in the current directory."""
        try:
            if not os.path.exists(self.keyword_filepath):
                warnings.warn(f"Keyword file not found in the current working directory: {self.keyword_filepath}", UserWarning)
                return None

            with open(self.keyword_filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            warnings.warn(f"Error decoding JSON from the file: {e}", UserWarning)
        except IOError as e:
            warnings.warn(f"IO error occurred while reading the file: {e}", UserWarning)
        return None
            



    def _create_property_regex(self, property_name):

        pattern = re.sub(r'\s+', r'\\s*-?\\s*', property_name.lower())
        return re.compile(r'\b' + pattern + r'\b', re.IGNORECASE)
        


    def _build_prompt(self):
        """Builds an interactive and conditional analysis prompt for extracting material properties from scientific texts."""



        def generate_property_pattern(keyword):
            parts = re.split(r'[- ]', keyword)  
            pattern = r'\b'  
            for part in parts:
                pattern += f'{part}[ -]?'  
            pattern += r's?\b' 
            return pattern


        property_pattern = generate_property_pattern(self.property)
        property_regex = re.compile(property_pattern, re.IGNORECASE)
     

        additional_instructions = self.__additional_instructions() #self.additional_prompts if self.additional_prompts else "No additional instructions provided."
        formula_standardization_checks = self._formula_standardization()
        
        standardized_formulas = self.get_standardized_formulas()
        
        system_prompt = f'''
        TASK DESCRIPTION:
        Analyze scientific literature to identify materials and their corresponding property related to the "{self.property}". Focus on recognizing all variations of the "{self.property}" and explicitly stated numerical values and 
        optionally, the method used in obtaining the "{self.property}" value (e.g., measured using AFM method). If method is not provided report "None".

        ADDITIONAL INSTRUCTIONS:
        {additional_instructions}  
        
        FORMULA STANDARDIZATION:
        {formula_standardization_checks}{self._unit_conversion()}\n\n
        Use the list below as a sample of how to convert material names and complex structures to standardized formulas: {standardized_formulas}

        PROCESS FLOW:
        1. Strictly confirm the presence of the primary keyword "{self.property}":
          - Check the text and tables for "{self.property}" using regex patterns and structured data extraction techniques. Be explicit to capture any variations (case variation, plural form, abreviation, etc.).
          - If "YES" ("{self.property}" found), proceed to STEP 2.
          - If "NO" ("{self.property}" not found), log the absence and halt further analysis for this text.

        2. Extract values, units, and methods associated with "{self.property}":
          - Identify all instances where "{self.property}" and its numerical values with units appear in the current text, ignoring any inferred or unrelated data.
          - (Be very strict): Check and verify if "{self.property}" is listed in a table. [Respond with "Yes" or "No" only]. If "Yes", focus specifically on columns that list "{self.property}" explicitly or implicitly as headers. Extract "{self.property}" and the unit from corresponding columns using contextual clues within the table.
          - Separate entries for each material and "{self.property}" value.
          - (Be strict). Is the [extracted value] of "{self.property}" for the given material correct? Be very strict. [Answer "Yes" or "No" only]. If "No", halt processing present text. 
          - (Be strict). If [extracted value] of "{self.property}" has [error value] to it, [e.g., 13 ± 1], get the average; if "{self.property}" is given as range, [e.g., 1-3.5], create separate entries for each range


	3. Strict Verification of Property Values, Keywords, and Chemical Formulae:
   	  - Keyword Association Check: Confirm whether the extracted "{self.property}" value is associated with any of the keywords "{self.keywords_str}" in the text. Consider the weights, which indicate importance and synonyms of the keywords:
              - "{self.keywords_str}"
   	  - Accuracy Confirmation (Be very strict): Verify if the "{self.property}" value [extracted value] is accurate for the identified material in this text. [Respond with "Yes" or "No"]. A "No" response requires re-evaluation of 
   	    the data extraction process.
   	  - Chemical Formula Validation (Be very strict): Ensure the identified chemical formula for the material matches the "{self.property}" value in this text. Is the formula correct? [Respond with "Yes" or "No"]. If "No", 
   	    reassess and validate the formula accuracy.
   	  - Unique Entry Creation (Be very strict): You seem to have created an entry for non-existing "{self.property}" and [chemical formula] in the present text. Do not create such entry.
   	  - Uniqueness and IUPAC Standards Compliance: Does the extracted material have a chemical formula that is uniquely identifiable and compliant with IUPAC standards? Respond with "Yes" or "No". If "No", skip creating entry 	
   	    for this material 
   	    
        
        4. Strict Validation of Compound Associations, Property Values, and Units:
          - Confirm the compound and "{self.property}" association: "There is a possibility that the data you extracted is incorrect. [Answer 'Yes' or 'No' only.] Be very strict. Is ',' the ',' compound for which the value of +"{self.property}"+ is given in the following text? Make sure it is a real compound."
          - Validate the value and "{self.property}": "There is a possibility that the data you extracted is incorrect. [Answer 'Yes' or 'No' only.] Be very strict. Is ',' the value of the "{self.property}" for the ',' compound in the following text?" If "No", halt processing present text. 
          - Verify the unit of "{self.property}": "There is a possibility that the data you extracted is incorrect. [Answer 'Yes' or 'No' only.] Be very strict. Is ',' the unit of the ',' value of "{self.property}" in the following text?" If "No", halt processing present text. 


            
        5. Compile a structured entry for each verified material, property value, original unit, and method:
        - Convert "{self.property}" values strictly to "{self.unit_conversion}".
        - Format entry as: Material [Chemical formula], "{self.property} (Converted): [value] (Original Unit: [original unit])", Method [method]
        
        
        EXAMPLES:
        - Correct: Material: Material, "{self.property} (Converted): value (Original Unit: {self.unit_conversion})", Method: Method
        - Incorrect (missing unit): Material: Material, "{self.property}: value", Method: Method
                
        REMINDER:
        (Be very strict): Ensure all information is accurate and complete. Exclude any material with incomplete or unclear data.
        '''

        system_prompt += "\n\n" + "\n\n".join(self.recent_analyses[-3:])
        return system_prompt

        

    def get_standardized_formulas(self):
        """Return a list of standardized chemical formulas."""
        material_dict = {
            'molybdenum disulfide': 'MoS2',
            'graphene': 'C',
            'quartz': 'SiO2',
            'polystyrene': '(C8H8)n',
            'boron nitride': 'BN',
            'hexagonal boron nitride': 'BN',
            'carbon nanotube': 'CNT',
            'tungsten disulfide': 'WS2',
            'black phosphorus': 'BP',
            'silicon carbide': 'SiC',
            'silicon nitride': 'Si3N4',
            'titanium dioxide': 'TiO2',
            'zinc oxide': 'ZnO',
            'cadmium selenide': 'CdSe',
            'h-BN': 'BN',
            'cubic boron nitride': 'BN',
            'lead sulfide': 'PbS',
            'aluminum oxide': 'Al2O3',
            'magnesium oxide': 'MgO'
        }

        composite_patterns = [
            ('BN/graphene', 'BNC'),
            ('BN/graphene/MoS2', 'BNCMoS2'),
            ('BN/graphene/BN', 'BNCBN'),
            ('TiO2/Pt/TiO2', 'TiO2PtTiO2')
        ]

        formulas = list(material_dict.values())
        formulas.extend([formula for _, formula in composite_patterns])
        return formulas

        
    def _build_keywords(self):
        """Construct keywords"""
        keyword_descriptions = []
        for keyword in self.keywords.get("keywords", []):
            description = f'"{keyword}" (Weight: {self.keywords.get("keyword_weights", {}).get(keyword, "high")})'
            synonyms = self.keywords.get("keyword_synonyms", {}).get(keyword, [])
            if synonyms:
                synonyms_formatted = ', '.join([f'"{syn}"' for syn in synonyms])
                description += f", including synonyms like {synonyms_formatted}"
            keyword_descriptions.append(description)
        keywords_str = "; ".join(keyword_descriptions)
        return keywords_str




    def _formula_standardization(self):
        """
        Returns a string containing guidelines for standardizing chemical formulas.
        This can be used to help users of the system understand how to format their chemical data entries.
        """
        return '''
            CHEMICAL FORMULA STANDARDIZATION:

            - Standardize simple compounds by removing dashes and other non-essential characters. 
              For example, "Al-Se" should be written as "AlSe" and "Ti-6Al-4V" should be "Ti6Al4V".

            - Adjust prefixes like "h-BN" to their standard chemical formula "BN".

            - Convert common material names to their respective chemical formulas for uniformity. 
              For example: 'Quartz' should be noted as 'SiO2', 'Graphite' as 'C', and 'Polystyrene' as '(C8H8)n'. 
              This ensures consistency in naming and notation across various texts.

            - For composite materials such as layered or mixed compounds, concatenate the individual components without slashes or spaces. 
              Provide a standardized chemical formula for each component in the sequence they appear. 
              For example:
                - "BN/graphene/BN" should be standardized to "BNCBN".
                - Similarly, "TiO2/Pt/TiO2" should be written as "TiO2PtTiO2".

            - Ensure all chemical formulas are presented without any spaces or special characters unless they denote a significant aspect of the chemical structure 
              (e.g., parentheses in polymers like (C8H8)n).

            This standardization aids in maintaining consistency and clarity in reporting and analyzing chemical data. Ensure each entry adheres to these rules to improve data uniformity and readability.
        '''
        

    def _unit_conversion(self):
        return f'''
        \n\nUNIT CONVERSION:\n\n
        (Be very strict): Convert and standardize the extracted "{self.property}" value to the unit, "{self.unit_conversion}". Record and note the "original unit" for each value to ensure accurate reporting. 
        (Be very strict): Verify and validate that the converted "{self.property}" value is correct. For example, you cannot be reporting that length has a unit of energy. 
        (Be very strict): Verify that you have recorded the "original unit" along with its extracted "{self.property}" value.
        '''

    def __additional_instructions(self):
        """Adds additional custom instructions if available."""
        if self.additional_prompts:
            return (f"\n\nEXAMPLES FOR ILLUSTRATION ONLY: Below are examples illustrating various ways "
                    f" '{self.property}' values and related properties might be described in texts. "
                    f"These examples are for understanding context and format only. "
                    f"Do not extract, report or include any properties from these examples in your analysis. "
                    f"(Be very strict): Verify and validate that you have only used this information to understand content in the present text, and not as data sources for extraction:\n"
                    f"{self.additional_prompts}\n\n"
                    )
        return ""


    def check_consistency(self, response_text):
        blob = TextBlob(response_text)
        properties = {}

        for sentence in blob.sentences:
            sentiment = sentence.sentiment.polarity
            for keyword in self.keywords:
                if keyword in sentence.lower():
                    if keyword in properties:
                        if (properties[keyword] > 0 and sentiment < 0) or (properties[keyword] < 0 and sentiment > 0):
                            return False
                    properties[keyword] = sentiment
        return True



    def check_relevance(self, response_text):
        doc = nlp(response_text)
        relevant_terms = set(['material', self.property] + self.keywords)

        found_terms = set()
        for token in doc:
            if token.lemma_ in relevant_terms:
                found_terms.add(token.lemma_)

        return len(found_terms) >= len(relevant_terms) / 2


    def get_element_data(self, symbol):
        """Fetch element data dynamically and cache it."""
        if symbol not in self.elements:
            try:
                self.elements[symbol] = get_element(symbol)
            except ValueError:
                print(f"Error: Element {symbol} not found.")
                return None
        return self.elements[symbol]


    def validate_chemical_formula(self, text):
        pattern = r'\b([A-Z][a-z]{0,2}(\d{0,3})?)+\b|\(\b([A-Z][a-z]{0,2}(\d{0,3})?)+\b\)(\d+)'
        matches = re.finditer(pattern, text)
        valid_formulas = []

        def parse_formula(formula):
            content = {}
            element_pattern = r'([A-Z][a-z]?)(\d*)'
            multiplier = 1
            if '(' in formula and ')' in formula:
                match = re.match(r'\(([^)]+)\)(\d+)', formula)
                if match:
                    formula = match.group(1)
                    multiplier = int(match.group(2))
            for element_match in re.finditer(element_pattern, formula):
                element_symbol, count = element_match.groups()
                count = int(count) if count else 1
                count *= multiplier


                element_data = self.get_element_data(element_symbol)

                if element_data is None:
                    #print(f"Element {element_symbol} not recognized.")
                    return False

                if element_symbol in content:
                    content[element_symbol] += count
                else:
                    content[element_symbol] = count

            return True


        for match in matches:
            formula = match.group(0)
            if parse_formula(formula):
                valid_formulas.append(formula)

        return valid_formulas


    def preprocess_abstract(self, abstract, diagnostics=False):
        if diagnostics:
            print("Original Abstract:")
            print(abstract)

        clean_abstract = re.sub(r'\s{2,}', ';', abstract)  # Replace multiple spaces with semicolon
        clean_abstract = re.sub(r'\(\?\)', '', clean_abstract)  # Remove uncertainty notations

        # Remove HTML/XML tags
        no_tags = re.sub(r'<[^>]+>', '', clean_abstract)

        if diagnostics:
            print("\nAfter Removing Tags:")
            print(no_tags)
            
        protected_formulas = re.sub(r'(\b[A-Za-z0-9]+(?:\/[A-Za-z0-9]+)+\b)', lambda x: x.group(0).replace('/', '∕'), no_tags)

        # Split the cleaned text into sentences
        sentences = sent_tokenize(protected_formulas)

        if diagnostics:
            print("\nAfter Sentence Tokenization:")
            for i, sentence in enumerate(sentences):
                print(f"Sentence {i+1}: {sentence}")

        # Join sentences to form a continuous block of text
        continuous_text = ' '.join(sentences)

        # Read keywords from the file specified by self.prep_keyword_path
        try:
            with open(self.prep_keyword_path, "r") as f:
                try:
                    keywords = json.loads(f.read())
                except json.JSONDecodeError:
                    f.seek(0)
                    keywords = [line.strip() for line in f.readlines()]
            
            # Create a pattern for the keywords
            pattern = '|'.join(keywords)

            # Filter the continuous text using the keywords
            if re.search(pattern, continuous_text, re.IGNORECASE):
                return continuous_text
            else:
                return None
        except FileNotFoundError:
            print(f"Error: '{self.prep_keyword_path}' file not found.")
            return None


    def build_history(self):
        if not self.recent_analyses or len(self.recent_analyses) < 3:
            return []  
        
        property_description = getattr(self, 'property', 'specific properties')
        history = [
            {
                'role': 'user',
                'parts': [f'Use these as examples on how to extract the material chemical formulas and {property_description} from the text accurately:']
            },
            {
                'role': 'model',
                # Construct the model part using the last three entries from recent_analyses
                'parts': [', '.join(self.recent_analyses[-3:])]
            }
        ]

        return history


    def retry_with_backoff(self, func, max_retries=5, backoff_in_seconds=1):
        """Retry a function with exponential backoff."""
        attempt = 0
        while attempt < max_retries:
            try:
                return func()
            except ReadTimeoutError as e:
                logging.warning(f"Timeout occurred, waiting {backoff_in_seconds * (2 ** attempt)}s before retrying...")
                time.sleep(backoff_in_seconds * (2 ** attempt))
                attempt += 1
            except requests.exceptions.RequestException as e:
                logging.error(f"An error occurred: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_in_seconds * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Non-timeout error, retrying in {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                    attempt += 1
                else:
                    logging.error("Max retries reached or critical error occurred.")
                    break
            except Exception as e:
                wait = backoff_in_seconds * (2 ** attempt) + random.uniform(0, 1)
                logging.error(f"Error: {e}. Retrying in {wait:.2f} seconds...")
                time.sleep(wait)
                attempt += 1
        raise Exception(f"Failed after {max_retries} retries.")

                           
    @backoff.on_exception(backoff.expo, ReadTimeoutError)   
    def analyze_abstract(self, abstract, max_retries=5, backoff_factor=1.5, timeout_wait=121, max_length=int(2021)):
        abstract = self.preprocess_abstract(abstract)
        current_part = ""
        parts = []
        history = self.build_history()

        sentences = re.split(r'(?<=[.!?]) +', abstract)
        for sentence in sentences:
            if len(current_part) + len(sentence) + 1 > max_length:
                parts.append(current_part)
                current_part = sentence
            else:
                current_part += (' ' + sentence if current_part else sentence)
        if current_part:
            parts.append(current_part)
        
        combined_result = ''

        def make_request(part):
            dynamic_prompt = f"{self._build_prompt()}\n\nNext text:\n{part}"
            response = None

            if self.model_type == 'gemini':
                chat = self.model.start_chat(history=history)
                response = chat.send_message(dynamic_prompt, generation_config=self.generation_config)

                if hasattr(response, 'parts'):
                    analysis_result = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                else:
                    raise ValueError("Unexpected API response format")

            elif self.model_type == 'chatgpt':
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": dynamic_prompt},
                        {"role": "user", "content": part}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                analysis_result = response.choices[0].message.content if response.choices else "Analysis failed."
            else:
                raise ValueError("Unsupported model type provided. Use 'gemini' or 'gpt-4'.")
            
            return analysis_result

        for part in parts:
            try:
                analysis_result = self.retry_with_backoff(lambda: make_request(part), max_retries=max_retries, backoff_in_seconds=backoff_factor)
                if self.check_consistency(analysis_result):  # and self.validate_chemical_formula(analysis_result): #and self.check_relevance(analysis_result): #
                    combined_result += analysis_result.strip() + " "
                else:
                    logging.warning("Formula verification failed.")
                    break
            except Exception as e:
                logging.error(f"Error during analysis: {e}")
                continue  # Move to the next part if one fails
                    
        if combined_result:
            self.recent_analyses.append(combined_result.strip())
            
        return combined_result.strip()


    def parse_labelled(self, line):

        pattern1 = re.compile(
            rf"material:\s*([\w\s]+?),\s*\"?{self.property}\"?\s*\(Converted\):?\s*\"?(\d+\.?\d*|None)\"?\s*\(Original Unit: (\w+|None)\),\s*Method:\s*(.*)",
            re.IGNORECASE
        )

        # Second pattern: Handles cases where the entire property descriptor is strictly quoted
        pattern2 = re.compile(
            rf"material:\s*([\w\s]+?),\s*\"{self.property} \(Converted\): (\d+\.?\d*|None) \(Original Unit: (\w+|None)\)\"\s*,\s*Method:\s*(.*)",
            re.IGNORECASE
        )



        # Try to match using the first pattern
        try:
            match = pattern1.search(line)
            if match:
                material, property_value, original_unit, method = match.groups()
                material = material.strip() if material else None
                property_value = None if property_value == "None" else property_value
                original_unit = None if original_unit == "None" else original_unit.strip()
                method = method.strip() if method else "None"
                return material, property_value, original_unit, method
        except ValueError:
            pass  # If no match, pass to the second pattern

        # Try to match using the second pattern
        match = pattern2.search(line)
        if match:
            material, property_value, original_unit, method = match.groups()
            material = material.strip() if material else None
            property_value = None if property_value == "None" else property_value
            original_unit = None if original_unit == "None" else original_unit.strip()
            method = method.strip() if method else "None"
            return material, property_value, original_unit, method
        else:
            raise ValueError("Labelled format not matched")



    def parse_delimited(self, line):
        if not line.strip() or re.match(r"^\|\s*(-+|\s+)\s*\|", line.strip()):
            return None 
        
        if re.match(r"\|\s*[\w\s]+?\s*\|\s*[^\d\.]+\s*\|\s*[\w\s]+?\s*\|", line):
            return None  # Ignore header lines

        pattern = re.compile(rf"\|\s*([\w\s]+?)\s*\|\s*(\d+\.?\d*)\s*\|\s*([\w\s]+?)\s*\|", re.IGNORECASE)
        match = pattern.search(line)
        if match:
            return match.groups() + (None,)  

  

    def parse_simple(self, line):
        """
        Parse lines that directly give the property value without conversion or original unit details.
        Example input: "Material: InSeI, Band gap: 2.12, Method: None"
        """
        # Regular expression to match the format without conversion details
        pattern = re.compile(
            rf"material:\s*([\w\s]+?),\s*{self.property}:\s*(\d+\.?\d*),\s*Method:\s*(.*)",
            re.IGNORECASE
        )
        match = pattern.search(line)
        if match:
            material, property_value, method = match.groups()
            default_unit = "unknown"  # You might have a default or you may leave it as None
            method = method if method else "None"  # Default to "None" if method is not specified
            return material, property_value, default_unit, method
        else:
            raise ValueError("Simple format not matched")


    def parse_new_format(self, line):
        pattern1 = re.compile(
            r"\*\*Material\*\*:\s*([\w\s]+)\n\s*-\s*\*\*Thickness \(Converted\)\*\*:\s*([\d\.]+\s*Å\s*to\s*[\d\.]+\s*Å)\s*\(Original Unit:\s*(\w+)\)\n\s*-\s*\*\*Method\*\*:\s*(.*)",
            re.IGNORECASE
        )

        # Second pattern: Handles single values (e.g., 13 Å)
        pattern2 = re.compile(
            r"\*\*Material\*\*:\s*([\w\s]+)\n\s*-\s*\*\*Thickness \(Converted\)\*\*:\s*([\d\.]+\s*Å)\s*\(Original Unit:\s*(\w+)\)\n\s*-\s*\*\*Method\*\*:\s*(.*)",
            re.IGNORECASE
        )
        try:
            match = pattern1.search(line)
            if match:
                material, thickness, original_unit, method = match.groups()
                material = material.strip() if material else None
                thickness = thickness.strip() if thickness else None
                original_unit = original_unit.strip() if original_unit else None
                method = method.strip() if method else "None"
                return material, thickness, original_unit, method
        except ValueError:
            pass

        match = pattern2.search(line)
        if match:
            material, thickness, original_unit, method = match.groups()
            material = material.strip() if material else None
            thickness = thickness.strip() if thickness else None
            original_unit = original_unit.strip() if original_unit else None
            method = method.strip() if method else "None"
            return material, thickness, original_unit, method
        else:
            raise ValueError("New format not matched")

        

    def parse_analysis_results(self, analysis_result):
        valid_entries = []
        lines = analysis_result.split('\n')
        for line in lines:
            result = None
            try:
                result = self.parse_labelled(line)
            except ValueError:
                pass  

            if not result:
                try:
                    result = self.parse_delimited(line)
                except ValueError:
                    pass  


            if not result:
                try:
                    result = self.parse_simple(line)
                except ValueError:
                    pass
                    

            if not result:
                try:
                    result = self.parse_new_format(line)
                except ValueError:
                    continue    
                                                           
            if result:
                material, property_value, original_unit, method = result
                if property_value is not None:
                    property_value = property_value.strip()
                if original_unit is not None:
                    original_unit = original_unit.strip()
                if method is not None:
                    method = method.strip()

                if material and material.strip() and property_value and property_value.strip():      
                    valid_entries.append((material, property_value, original_unit, method))

        return valid_entries



    def read_file_with_detected_encoding(self, filepath, sample_size=4096):
        """Read a file using detected encoding, with fallback and error handling."""
        try:
            with open(filepath, 'rb') as file:
                raw_data = file.read(sample_size)
                result = detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                logging.info(f"Detected encoding {encoding} with confidence {confidence}")

            if encoding and confidence > 0.5:  
                try:
                    return pd.read_csv(filepath, encoding=encoding)
                except UnicodeDecodeError:
                    #logging.error(f"Failed to decode file {filepath} with detected encoding {encoding}. Trying UTF-8.")
                    return pd.read_csv(filepath, encoding='utf-8')  # Fallback to UTF-8
            else:
                #logging.info(f"Low confidence in detected encoding. Trying UTF-8 as fallback.")
                return pd.read_csv(filepath, encoding='utf-8')
        except Exception as e:
            #logging.error(f"Unhandled error while reading file {filepath}: {e}")
            raise ValueError(f"Failed to read the file {filepath} with any known encoding.") from e
       
            
    def process_and_save_text(self, input_csv, text_column, output_csv_prefix):   #='materialppt'):
        file_extension = os.path.splitext(input_csv)[-1].lower()


        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = output_csv_prefix    #f"{output_csv_prefix}_{timestamp}.csv"


        if os.path.exists(output_csv):
            os.remove(output_csv)
            logging.info(f"Existing file {output_csv} removed.")

        try:
            if file_extension == '.csv':
                examples = self.read_file_with_detected_encoding(input_csv) #pd.read_csv(input_csv) #
            elif file_extension in ['.xls', '.xlsx']:
                examples = pd.read_excel(input_csv)
            else:
                message = "Unsupported file format"
                logging.error(message)
                print(message)
                return
        except Exception as e:
            message = f"Error reading the file: {e}"
            print(message)
            logging.error(message)
            
            return


        examples.dropna(subset=[text_column], how='all', inplace=True)

        file_exists = os.path.isfile(output_csv)

       

        for index, row in examples.iterrows():
            try:
                abstract = row[text_column]

                logging.info(f"Analyzing text {index + 1}/{len(examples)}")
                analysis_result = self.analyze_abstract(abstract)
                #print("analysis_result", analysis_result)
                parsed_results = self.parse_analysis_results(analysis_result)
                #print("parsed_results ", parsed_results)
                



                #if not parsed_results:
                #    logging.info(f"No valid data found for abstract {index + 1}. Skipping...")
                #    continue
                
                with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([f"Timestamp: {timestamp}"])
                        writer.writerow(["MaterialName", f"{self.property.capitalize()}({self.unit_conversion})", "OriginalUnit", "Method"])
                        file_exists = True

                    for material, property_value, unit, method in parsed_results:
                        #if material and property_value:  
                        if material is not None and material.strip() and property_value is not None and property_value.strip():
                            writer.writerow([material, property_value, unit, method])
                            logging.info(f"Material: {material}, {self.property}: {property_value} (Unit: {unit})")
                        else:
                            logging.warning(f"Incomplete data for abstract {index + 1}. Skipping entry.")
            except Exception as e:
                logging.error(f"An error occurred processing abstract {index + 1}: {e}")

