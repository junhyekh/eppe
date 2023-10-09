# FER Occupancy Network implementation

## Setup environment
Setup conda environment using...
```bash
conda env create --file environment.yaml
```
and install local packages
```bash
python setup.py build_ext --inplace
```

## Dataset
- Download preprocessed ShapeNet from [link](https://github.com/autonomousvision/occupancy_networks) and extract it to `data/ShapeNet`
- Download preprocessed ShapeNet rotation file from [link](https://drive.google.com/file/d/13VQRV7vJojhUhE-nssEXlH2oI6vzMPwy/view?usp=drive_link) and extract it to `data_rotations/ShapeNet/*`
- Download preprocessed EGAD from [link](https://drive.google.com/file/d/1oAipEm2U77F0a9t_g5CcPyoit98BaAXt/view?usp=drive_link) and extract it to `data/EGAD/*`.
- Download preprocessed EGAD rotation file from [link](https://drive.google.com/file/d/1So54FBaBRo4N9Li5ulb1zsna0ZpTxjPq/view?usp=drive_link) and extract it to `data_rotations/EGAD`.
- Download pretrained weights from [link](https://drive.google.com/file/d/17fdWjUVMPjWIEkDFTWAx9Ir5GCbhiM1A/view?usp=drive_link) and put under `results/*`


## Training
```bash
python train_evn.py --config configs/equinet/evn_pointnet_resnet_aligned.yaml 
python train_evn.py --config configs/equinet/evn_pointnet_resnet_so3.yaml 
python train_evn.py --config configs/egad/egad_evn_pointnet_resnet_so3.yaml 
python train_evn.py --config configs/egad/egad_evn_pointnet_resnet_so3.yaml 
```

# Shape evaluation
```bash
python eval_evn.py --config configs/equinet/evn_pointnet_resnet_aligned.yaml 
python eval_evn.py --config configs/equinet/evn_pointnet_resnet_so3.yaml 
python eval_evn.py --config configs/egad/egad_evn_pointnet_resnet_so3.yaml 
python eval_evn.py --config configs/egad/egad_evn_pointnet_resnet_so3.yaml 
```

# Registration evaluation
```bash
python registration_eval_evn.py --config configs/registration/evn_pointnet_resnet_registration.yaml 
```