# ContraCheck

## Overview

ContraCheck aims to mine social media data, preferably Twitter, and identify tweets made in matching contexts and highlight them if they are of a contradictory nature. ContraCheck fetches tweets using a web scraper and utilizes a classifier to compare them, and highlight such tweets as neutral, contradictions or entailments.

## Setting up the environment

Python 3.0 and above are required to be installed in the environment. 

Additional dependencies as follows are required to be installed using the `pip install` command:

- tensorflow 1.2
- keras 2.0
- scipy
- numpy
- pandas
- scikit-learn

## Getting started

Once Python and all the dependencies are set up, you will need to provide datasets for the classifier to train and validate. These are available in the `Training Data` folder. Copy the text files provided in the folder to the root directory.

Then, open up your terminal/working environment and:

- to grab the tweets, run `python tweetgrabber.py`
- to train the classifier, run `python model2.py`
- to run the classifier against the provided data, run `python predict.py`

## Acknowledgements

This project has been developed as part of our Information Retrieval (CS-309) course under Prof. [Ashish Sureka](http://www.ashish-sureka.in) at [Ashoka University](http://ashoka.edu.in).

This project has been created with love and care by Simran Bhuria, Vijay Lingam, Mayukh Nair and Divij Singh.

The classifier in this project was trained on the following datasets: 

- [PHEME RTE Dataset](https://www.pheme.eu/2016/04/12/pheme-rte-dataset/)
- [Stanford Contradictory Corpora](https://nlp.stanford.edu/projects/contradiction/)
