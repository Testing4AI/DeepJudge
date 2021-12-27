## To run 

#### Step1: Select seeds for a trained model (the owner model)
```python
$ python seed_selection.py --parameters
```
This will create a `seeds` directory and save the selected seeds. 


#### Step1: Generate test cases (black-box and white-box)

```python
$ python blackbox_generation.py --parameters
$ python whitebox_generation.py --parameters
```
This will create a `testcases` directory and save the generated testcases. 


#### Step2: Metric evaluations 

```python
$ python blackbox_evaluation.py --parameters
$ python whitebox_evaluation.py --parameters
```
This will create a `results` directory and save the evaluation results. 
