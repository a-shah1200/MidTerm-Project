# LLM Text Completion Classifier

## Project Overview

This project focuses on building a deep learning classifier to identify which Large Language Model (LLM) generated a given text completion. Starting with a human-written text fragment (denoted as `x_i`), such as "Yesterday I went," different LLMs are used to complete the sentence with varying continuations (`x_j`), for example, "to Costco and purchased a floor cleaner." The goal of the classifier is to recognize the specific patterns, stylistic, linguistic, or structural differences in the generated completions, allowing it to classify which LLM produced the continuation.

## Project Structure

### Datasets

The project includes a multi-step process for generating and preparing datasets:

1. **Text Extraction and Sentence Generation**: The process begins with human-written fragments (e.g., `x_i`).
2. **Dataset Cleaning**: Removal of irrelevant punctuation and unnecessary characters from the extracted text.
3. **LLM Completions**: Different LLMs generate continuations of the human-written fragments, resulting in several datasets based on each LLM's output.
4. **Final Dataset**: The LLM-generated completion datasets are concatenated to create `LLM_dataset.csv`, which is used for training the classifier.

The relevant scripts and notebooks for these steps can be found in the following directories:

- `datasets/`: Contains all intermediate and final datasets, including the `incomplete_dataset.csv`, `cleaned_dataset.csv`, and LLM completion datasets for models like GPT2, BART, Pegasus, etc. The final dataset for classifier training is `LLM_dataset.csv`.

### Classifier Models

We trained the classifier using three different pre-trained models:

1. **ALBert Classification**
2. **DistilBert Classification**
3. **Roberta-Peft Classification**

Each model has its own folder containing the implementation notebook and evaluation results:

- **ALBert_Classification**: Includes the model code, results, and evaluation metrics.
- **DistilBert_Classification**: Contains the model and evaluation files, with updated paths and results.
- **Roberta-Peft**: Provides the implementation and corresponding evaluation for the RoBERTa model.

### Important Files

- `dataset_creation.ipynb`: A notebook that creates the dataset by extracting text, cleaning it, and generating the final training dataset.
- `Roberta-Peft/classifier_final_version.ipynb`:  A notebook that for training and testing RoBERTa model with LoRA technique.
- `DistilBert_Classification/Classifier.ipynb`: A notebook that for training and testing DistilBERT model.
- `ALBert_Classification/ALBert_Classifier.ipynb`: A notebook that for training and testing ALBERT model.


### Results

Each model's results, including accuracy, precision, recall, and F1-score, are displayed within their respective folders. Additionally, various graphs and visualizations for the model performances are provided.

## Usage

1. **Dataset Creation**: Run the `dataset_creation.ipynb` notebook to create and preprocess the dataset.
2. **Model Training**: Execute the classification notebooks as specified above to train and evaluate the respective classifiers.
3. **Evaluation**: Visualize and compare the results to determine which model best classifies the LLM-generated text completions.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE License - see the `LICENSE` file for details.
