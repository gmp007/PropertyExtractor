# PropertyExtractor: An Open-Source Conversational LLM-Based Tool

## Introduction

The advent of natural language processing and large language models (LLMs) has revolutionized the extraction of data from unstructured scholarly papers. However, ensuring data trustworthiness remains a significant challenge. **PropertyExtractor** is an open-source tool that leverages advanced conversational LLMs like **Google Gemini Pro** and **OpenAI GPT-4**, blends zero-shot with few-shot in-context learning, and employs engineered prompts for the dynamic refinement of structured information hierarchies to enable autonomous, efficient, scalable, and accurate identification, extraction, and verification of material property data to generate material property database. 

## Features

- **Advanced LLM Integration**: Supports both Google Gemini Pro and OpenAI GPT-4.
- **Zero-shot and Few-shot Learning**: Blends in-context learning for better extraction accuracy.
- **Engineered Prompts**: Dynamic refinement of structured information hierarchies.
- **Autonomous Extraction**: Efficient and scalable identification and extraction of material properties.
- **High Precision and Recall**: Achieves over 90% precision and recall with an error rate of approximately 10%.

## Installation

**PropertyExtractor** offers straightforward installation options suitable for various user preferences as explained below. We note that all the libraries and dependables are automatically determined and installed alongside the PropertyExtractor executable **"propertyextract"** in all the installation options. 

1. **Using pip**: Our recommended way to install the **PropertyExtractor** package is using pip. 
   - Quickly install the latest version of the **PropertyExtractor** package with pip by executing: 
     ```
     pip install -U propertyextract
     ```

2. **From Source Code**:
   - Alternatively, users can download the source code with:
     ```
     git clone [git@github.com:gmp007/PropertyExtractor.git]
     ```
   - Then, install **PropertyExtractor** by navigating to the master directory and running:
     ```
     pip install .
     ```

3. **Installation via setup.py**:
   - PropertyExtractor can also be installed using the `setup.py` script:
     ```
     python setup.py install [--prefix=/path/to/install/]
     ```
   - The optional `--prefix` argument is useful for installations in environments like shared High-Performance Computing (HPC) systems, where administrative privileges might be restricted.
   - Please note that while this method remains supported, its usage is gradually declining in favor of more modern installation practices. We only recommend this installation option where standard installation methods like `pip` are not applicable.
   
## Usage

### Configuration

Please don't expose your API keys. Before running **PropertyExtractor**, configure the API keys for Google Gemini Pro and OpenAI GPT-4 as environment variables.

### On Linux/macOS

```bash
export GPT4_API_KEY='your_gpt4_api_key_here'
export GEMINI_PRO_API_KEY='your_gemini_pro_api_key_here'
```

### On Windows

```bash
set GPT4_API_KEY='your_gpt4_api_key_here'
set GEMINI_PRO_API_KEY='your_gemini_pro_api_key_here'
```
   
## Usage and Running PropertyExtractor

**PropertyExtractor** is easy to run. The key steps for initializing **PropertyExtractor** follows:

1. **Unstructured data generation***: Use API to obtain the material property that you want to generate the database from the publishers of your choice. We have written API functions for Elsevier's ScienceDirect API, CrossRef REST API, and PubMed API. We can share some of these if needed. 

2. **Create a Calculation Directory**:
   - Start by creating a directory for your calculations.
   - Run `propextract -0` to generate the main input template of the **PropertyExtractor**, which is the `extract.in`. Modify following the detailed instructions included.
   - Optional files such as the `additionalprompt.txt' for augmenting additional custom prompts and `keywords.json' for custom additional keywords to support the primary keyword are also generated. Modify to suit the material property being extracted. The main input template `extract.in' looks like below:
     ```
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
      inputfile_name = 2Dthickness_Elsevier.csv
      # Column name in the input file to be processed
      column_name = Text
      # Name of output file
      outputfile_name = ppt_test
     ```

3. **Initialize the Job**:
   - Execute `propextract` to begin the calculation process.

5. **Understanding PropertyExtractor Options**:
   - The main input file `extract.in` includes descriptive text for each flag, making it user-friendly.

## Citing PropertyExtractor
If you have used the **PropertyExtractor** package in your research, please cite:
  - [Dynamic In-context Learning with Conversational Models for Data Extraction and Materials Property Prediction](https://doi.org/xxxx) - 

```latex
@article{Ekuma2024,
  title = {Dynamic In-context Learning with Conversational Models for Data Extraction and Materials Property Prediction},
  journal = {XXX},
  volume = {xx},
  pages = {xx},
  year = {xx},
  doi = {xx},
  url = {xx},
  author = {Chinedu Ekuma}
}
```

```latex
@misc{PropertyExtractor,
  author = {Chinedu Ekuma},
  title = {PropertyExtractor -- LLM-based model to extract material property from unstructured dataset},
  year = {2024},
  howpublished = {\url{https://github.com/gmp007/PropertyExtractor}},
  note = {Open-source tool leveraging LLMs like Google Gemini Pro and OpenAI GPT-4 for material property extraction},
}
```



## Contact Information
If you have any questions or if you find a bug, please reach out to us. 

Feel free to contact us via email:
- [cekuma1@gmail.com](mailto:cekuma1@gmail.com)

Your feedback and questions are invaluable to us, and we look forward to hearing from you.

## License

This project is licensed under the GNU GPL version 3 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award DOE-SC0024099.

---

