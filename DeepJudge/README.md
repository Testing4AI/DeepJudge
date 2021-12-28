## To run 

#### Step1: Select seeds for a trained model (the owner model)
```python
$ python seed_selection.py --model owner_model_path --num 1000 
```

An example DNN: owner_model_path = `../train_models/attacks/example/cifar10_resnet20.h5` 

This will create a `seeds` directory and save the selected seeds. 


#### Step2: Generate test cases (black-box/white-box)

```python
$ python blackbox_generation.py --model owner_model_path --seeds seeds_path --method pgd --ep 0.03 --iters 10 
$ python whitebox_generation.py --model owner_model_path --seeds seeds_path --layer 3 
```
This will create a `testcases` directory and save the generated test cases. 


#### Step3: Metric evaluations (black-box/white-box)

```python
$ python blackbox_evaluation.py --model owner_model_path --suspect suspect_model_path --tests black_tests.npz
$ python whitebox_evaluation.py --model owner_model_path --suspect suspect_model_path --tests white_tests.npy
```
This will create a `results` directory and save the evaluation results. 
