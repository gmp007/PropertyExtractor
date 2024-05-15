# PropertyExtractor: An Open-Source Conversational LLM-Based Tool

## Introduction

The advent of natural language processing and large language models (LLMs) has revolutionized the extraction of data from unstructured scholarly papers. However, ensuring data trustworthiness remains a significant challenge. In this paper, we introduce **PropertyExtractor**, an open-source tool that leverages advanced conversational LLMs like **Google Gemini Pro** and **OpenAI GPT-4**, blends zero-shot with few-shot in-context learning, and employs engineered prompts for the dynamic refinement of structured information hierarchies. This enables autonomous, efficient, scalable, and accurate identification, extraction, and verification of material property data. **PropertyExtractor** can be used to autonomously generate any material property database.

## Features

- **Advanced LLM Integration**: Supports both Google Gemini Pro and OpenAI GPT-4.
- **Zero-shot and Few-shot Learning**: Blends in-context learning for better extraction accuracy.
- **Engineered Prompts**: Dynamic refinement of structured information hierarchies.
- **Autonomous Extraction**: Efficient and scalable identification and extraction of material properties.
- **High Precision and Recall**: Achieves over 90% precision and recall with an error rate of approximately 10%.

## Installation

**PropertyExtractor** offers straightforward installation options suitable for various user preferences as explained below. We note that in all the installation options, all the libraries and dependables are automatically determined and installed alongside the PropertyExtractor. 

1. **Using pip**: Our recommended way to install the **PropertyExtractor** package is using pip. 
   - Quickly install the latest version of the SMATool package with pip by executing: 
     ```
     pip install -U propextract
     ```

2. **From Source Code**:
   - Alternatively, users can download the source code with:
     ```
     git clone [git@github.com:gmp007/PropertyExtractor.git]
     ```
   - Then, install SMATool by navigating to the master directory and running:
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

Before running **PropertyExtractor**, configure the API keys for Google Gemini Pro and OpenAI GPT-4 in the `config.json` file:

```json
{
    "google_gemini_pro_api_key": "YOUR_GOOGLE_GEMINI_PRO_API_KEY",
    "openai_gpt4_api_key": "YOUR_OPENAI_GPT4_API_KEY"
}
```
   
## Usage and Running PropertyExtractor

The best way to learn how to use the SMATool package is to start with the provided examples folder. The key steps for initializing SMATool follows:

1. **Create a Calculation Directory**:
   - Start by creating a directory for your calculations.
   - Run `smatool -0` to generate the main input template of the SMATool, which is the `smatool.in`.

2. **Modify Input Files**:
   - Customize the generated files according to your project's requirements, choose the code type between VASP and QE, and specify the directory of your potential files. 

3. **Initialize the Job**:
   - Execute `propextract` to begin the calculation process.

4. **Understanding PropertyExtractor Options**:
   - The main input file `extract.in` includes descriptive text for each flag, making it user-friendly.

## Citing SMATool
If you have used the SMATool package in your research, please cite:
  - [SMATool: Strength of materials analysis toolkit](https://doi.org/10.1016/j.cpc.2024.109189) - 

@article{Ekuma2024,
  title = {SMATool: Strength of Materials Analysis Toolkit},
  journal = {Computer Physics Communications},
  volume = {300},
  pages = {109189},
  year = {2024},
  doi = {https://doi.org/10.1016/j.cpc.2024.109189},
  url = {https://www.sciencedirect.com/science/article/abs/pii/S0010465524001127},
  author = {Chinedu Ekuma}
}


## Have Questions
To join the Google user group, post your questions, and see if your inquiries have already been addressed, you can visit the [SMATools User Group](https://groups.google.com/g/smatools/) on Google Groups. This platform allows for interactive discussions and access to previously answered questions, facilitating a comprehensive support community.


## Contact Information
If you have any question or if you find a bug, please reach out to us. 

Feel free to contact us via email:
- [cekuma1@gmail.com](mailto:cekuma1@gmail.com)

Your feedback and questions are invaluable to us, and we look forward to hearing from you.

## License

This project is licensed under the GNU GPL version 3 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award DOE-SC0024099.

---
