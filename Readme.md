# Unsupervised Academic Concept Extraction

This is a C++ implementation of the Academic Concept Extraction framework as described in our paper,

<html>Adit Krishnan<sup>*</sup>, Aravind Sankar<sup>*</sup><html>, Shi Zhi and Jiawei Han [Unsupervised Concept Categorization and Extraction from Scientific Document Titles](https://arxiv.org/pdf/1710.02271.pdf) (CIKM 2017)

## Requirements
* g++

## PhraseType demo

```bash
g++ phraseType.cpp -o phraseType.o
./phraseType.o sample.txt
```

The probability distributions inferred by the Gibbs sampler will be written to the `PhraseType/` directory.

## DomainPhraseType demo

```bash
g++ domainPhraseType.cpp -o domainPhraseType.o
./domainPhraseType.o sample.txt
```
The probability distributions inferred by the Gibbs sampler will be written to the `DomainPhraseType/` directory.

## Data

In order to use your own data, you have to provide a text file of phrases with the following details for each phrase.  
* Left relation phrase - `lrp`
* List of words (comma-separated) `words`
* List of significant phrases (comma-separated) `sig_phrases`
* Right relation phrase `rrp`
* Venue `venue`

Each line in the text file contains a single input phrase of the format:

`lrp$words$sig_phrases$rrp$venue
`

For example, in a title `bootstrapped named entity recognition for product attribute extraction`,
the phrase `bootstrapped named entity recognition` may be given as:

`empty$bootstrapped,named,entity,recognition$entity_recognition$for$emnlp`

An efficient Pitman-Yor Adaptor Grammar implementation is available at http://web.science.mq.edu.au/~mjohnson/Software.htm.

## Cite

Please cite our paper if you use this code in your own work:

```
@article{krishnan2017unsupervised,
  title={Unsupervised Concept Categorization and Extraction from Scientific Document Titles},
  author={Krishnan, Adit and Sankar, Aravind and Zhi, Shi and Han, Jiawei},
  journal={arXiv preprint arXiv:1710.02271},
  year={2017}
}
```
