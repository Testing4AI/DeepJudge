## To run

#### Step1: Embed the watermark to the owner model (train from scratch) 

```python
$ python embed.py 
```
This will create a `logs` directory including training logs, the watermarked model and verification keys. 

#### Step2: Verify the watermark

```python
$ python evaluate.py 
```
This will calculate the TSA (trigger set accuracy) for the watermarked model. 
