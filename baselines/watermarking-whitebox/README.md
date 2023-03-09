## To run

#### Step1: Embed the watermark into a model (train from scratch) 

```python
$ python embed.py 
```
This will create a `logs` directory including a watermarked model and the verification keys. 


#### Step2: Verify the watermark

```python
$ python evaluate.py 
```
This will calculate the BER (bit error rate) for the watermarked model. 
