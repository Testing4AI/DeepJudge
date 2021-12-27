## To run

#### Step1: Generate test cases for a trained model (owner model)

```python
$ python generation.py 
```
This will saves an file `./key_xy.npz` (fingerprint). 

#### Step2: Metric evaluations between models 

```python
$ python evaluate.py 
```
This will calculate the MR (matching rate) on the pre-generated fingerprint. 
