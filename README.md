# Bacteria

## Environment

**torch == 2.0.1+cu118**
**python == 3.8.8**



## Dataset

All the data needed are available at  https://github.com/csho33/bacteria-ID.



## Training

If you want to load the pretrained model , please add "--load" in your command.

```python
python train.py --batch_size 32 --epochs 100 --device 0 --lr 0.0001 --ratio 0.9 --train_dir ./data/train --load
```



## Testing

Make sure you use the argument "--load " to load the trained model.

```python
python predict.py --batch_size 32 --device 0 --test_dir ./data/test --load 
```



## QuickStart

Run the demo.ipynb to start this project quickly.

## Ref
Chen, J., Hu, J., Xue, C. et al. Combined Mutual Learning Net for Raman Spectral Microbial Strain Identification. *Anal. Chem.* 96, 15 (2024). [https://doi.org/10.1021/acs.analchem.3c05107](https://doi.org/10.1021/acs.analchem.3c05107)

