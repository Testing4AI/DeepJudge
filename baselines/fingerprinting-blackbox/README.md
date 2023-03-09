## To run

#### Step1: Generate the fingerprint (a set of adversarial examples) for a trained model

```python
$ python generation.py 
```
This will save a file `key_xy.npz` (fingerprint).

#### Step2: Verify the fingerprint

```python
$ python evaluate.py 
```
This will calculate the MR (matching rate) on the pre-generated fingerprint. 

