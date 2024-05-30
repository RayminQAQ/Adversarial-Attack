# Adversarial-Attack
This project focuses on the development and analysis of Poison Attack and Evasion Attack strategies, crucial tools for understanding and enhancing the robustness of neural networks against adversarial threats.

## Project Structure

```
Repository file structure:
|-- Basic_pipline.py
|-- Poison_Attack.py
|-- Envasion_Attack.py
|-- model.py
|-- utils.py
|-- dataset.zip
|-- README.md
```

```
Your image dataset structure should be like:
|-- Image_directory
    |-- label1
        |-- images ...
    |-- label2
        |-- images ...
    |-- label3
        |-- images ...
    ...
    |-- labeln
```


## Pipeline
There are three piplines can use for machine learning training and testing:
1. **Basic_pipline.py**: Basic model without adding any Adversarial Attack method.
2. **Poison_Attack.py**: Based on Basic_pipline and add Poison Attack measure.
3. **Envasion_Attack.py**: Based on Basic_pipline and add Evasion Attack measure.

## Attack Strategies
### Evasion Attack
This type of attack occurs in the inference phase of the machine learning model. The attacker modifies the input data by adding slight differences that are not easily detectable by humans and inputs it into the AI system, causing the system to misclassify it. 

This attack does not affect the training phase. The attacker's goal is to make the machine learning model output incorrect results, while the input appears normal to human observers. This attack is commonly used in the field of image processing.

### Poisoning Attack
These attacks occur in the training phase of the machine learning model. The attacker injects malicious data (such as backdoors) into the training dataset, causing the model to make incorrect predictions when encountering certain types of inputs in the future.

If you want to use our dataset, please unzip: **dataset.zip**

## Result
We can achieve accuracy of **above 90%** in 10 epochs, using those pipline.

## Contributors
- **[RayminQAQ](https://github.com/RayminQAQ)**

If there's any question, contact: B11132009@mail.ntust.edu.tw
