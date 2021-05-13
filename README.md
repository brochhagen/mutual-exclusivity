# README
Code repository for Gulordava, Brochhagen &amp; Boleda (2020): [Deep daxes: Mutual exclusivity arises through both learning biases and pragmatic strategies in neural networks](https://cognitivesciencesociety.org/cogsci20/papers/0479/0479.pdf)

Get in touch if you have any questions!

*** 
### Requirements
The code is written in python 3

A conda environment named *me*, and supplied within, fulfills all the requirements to run the code.

To import and activate it, run:

```bash
conda env create --file environment.yaml #import environment from YAML
conda activate me 
```

***

### Data
The scripts expect the symbolic and visual data to be placed in appropriate `mutual-exclusivity/data`-(sub)folders

  * For example, the data for symbolic experiments using the transcriptions of CHILDES used in Frank et al. (2009) may be placed in `data/frank2009/all_words`;
  * and the pre-processed bounding boxes of Flickr30K objects in `data/flickr`

Refer to the paper for details

***


### Basic examples on how to run experiments
Results are written to `mutual-exclusivity/results`

#### Symbolic experiments
`python mutual-exclusivity/src/train_symbolic.py --data mutual-exclusivity/data/frank2009/all_words --seed 1008 --loss maxmargin_words`

#### Visual experiments
Evaluate on dogs (from Flickr training set) without competition 

`python  mutual-exclusivity/src/train_flickr.py --data mutual-exclusivity/data/flickr/ --debug --lr 0.1 --loss maxmargin_words --novel_set dogs`