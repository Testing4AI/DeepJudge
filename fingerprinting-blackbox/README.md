## To run

#### Step1: Generate a fingerprint (a set of examples) for the model

```python
$ python generation.py 
```
This will saves an file `./key_xy.npz` (fingerprint). 

#### Step2: Verify the fingerprint

```python
$ python evaluate.py 
```
This will calculate the MR (matching rate) on the pre-generated fingerprint. 
