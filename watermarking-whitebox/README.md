## To run

#### Step1: Embed the watermark to the owner model (train from scratch) 

```python
$ python embed.py 
```
This will creates a `logs` directory including training logs, the watermarked model and verification keys. 

#### Step2: Verify the watermark

```python
$ python evaluate.py 
```
This will calculate the BER (bit error rate) for the watermarked model. 
