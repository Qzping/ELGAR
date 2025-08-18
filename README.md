# ELGAR

The offical implementation of [ELGAR: <ins>**E**</ins>xpressive Ce<ins>**L**</ins>lo Performance Motion <ins>**G**</ins>eneration for <ins>**A**</ins>udio <ins>**R**</ins>endition]("https://arxiv.org/abs/2505.04203").

Visit our [project page](https://metaverse-ai-lab-thu.github.io/ELGAR/) for more demos.

Get the [SPD-GEN Dataset](https://forms.gle/pF7KMZ2wrnFQCbY87).

Visit our [last paper](https://metaverse-ai-lab-thu.github.io/String-Performance-Dataset-SPD/) for markerless motion capture with audio signals for string performance capture.

[![poster](assets/poster.jpg)](assets/poster.jpg)

## 0. Requirements
We test our code with Python 3.11 and PyTorch 2.3.1 (CUDA 12.1)

```
conda create -n ELGAR python=3.11
conda activate ELGAR
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Jukebox and Jukemirlib Installation
```
cd <your_path>/ELGAR/audio_encoder/jukebox
python setup.py bdist_wheel
pip install ./dist/jukebox-1.0-py3-none-any.whl
cd <your_path>/ELGAR/audio_encoder/jukemirlib
python setup.py bdist_wheel
pip install ./dist/jukemirlib-0.0.0-py3-none-any.whl
```

Download the Jukebox [vqvae cache]("https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar") and [prior cache]("https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar"). Place them in the following directory.

<details>
<summary>Jukebox Checkpoint Directory</summary>

```
ELGAR
|   ...
└───audio_encoder
│   └───model
│       |   vqvae.pth.tar
|       |   prior_level_2.pth.tar
```
</details>

### SMPL-X Download
Download SMPL-X v1.1 from the official [website]("https://smpl-x.is.tue.mpg.de/download.php"). Please note that the SMPL-X model has its own [license]("https://smpl-x.is.tue.mpg.de/modellicense.html"), which is different from the MIT license of this repository. Place the model in the following directory. (We use the neutral version of SMPL-X)

<details>
<summary>SMPL-X Model Directory</summary>

```
ELGAR
|   ...
└───model
│   ...
|   └───smplx
|       |   SMPLX_NEUTRAL.npz
|       |   ...
```
</details>

### VPoser Checkpoint Download
Download VPoser v2.0 from the official [website]("https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=V02_05.zip"). Please note that the VPoser has its own [license]("https://smpl-x.is.tue.mpg.de/modellicense.html"), which is different from the MIT license of this repository. Place the VPoser checkpoint in the following directory.

<details>
<summary>VPoser Checkpoint Directory</summary>

```
ELGAR
|   ...
└───data_process
|   └───ik_joints
|       └───human_body_prior
|           └───models
|               └───V02_05
|               |   ...
```
</details>

## 1. Let's Perform!
Download the ELGAR [checkpoint](https://forms.gle/pF7KMZ2wrnFQCbY87) and prepare your test audio. Place it in the following directory.

<details>
<summary>ELGAR Checkpoint and Test Audio Directory</summary>

```
ELGAR
|   ...
└───dataset
|   └───wild_data
|       └───cello
|           └───audio
|               └───test.wav (Place your audio here!)
└───save
│   └───full
```
</details>
<br>

Extract the test audio feature.
```
cd <your_path>/ELGAR/data_process
python prepare_inference_data.py --instrument cello --data_type wild --filename your_audio_name.wav
```

Run `bash test_script.sh` to generate the performance motion. (Modify the filename in the script)

## 2. Data Preparation
This step is to obtain the [SPD-GEN dataset](https://forms.gle/pF7KMZ2wrnFQCbY87) and the processed data for training.

Make sure you have downloaded the [SPD](https://forms.gle/pF7KMZ2wrnFQCbY87) and placed the data in the following directory.

<details>
<summary>SPD Data Directory</summary>

```
ELGAR
|   ...
└───dataset
│   └───SPD
│       └───cello
|           └───hand_rot
|           └───kp3d
```
</details>

### 2.1 Data Normalization
Run `python data_normalization.py` to get the normalized body in the following directory.

<details>
<summary>Normalized Data Directory</summary>

```
ELGAR
└───data_process
|   |   ...
│   │   data_normalization.py
│   └───train_data
│       └───wholebody_normalized
|           |   cello01.json
|           |   cello02.json
|           |   ...
|           |   cello85.json
```
</details>

### 2.2 Inverse Kinematics for Joint Rotations
To get the joint rotations of the normalized body:
```
python script_ik_joints.py --instrument cello --ik_time 1
```

Set the `ik_time` from 1 to 3 as our IK include 3 steps.

You will obtain the ik results in the following directory.

<details>
<summary>IK Result Directory</summary>

```
ELGAR
└───data_process
│   └───ik_joints
|       |   ...
|       |   script_ik_joints.py
│       └───results
|           |   body_cello01_round_1.pkl  # ik_time 1
|           |   body_cello01_round_2.pkl  # ik_time 2
|           |   body_cello01_calc.pkl     # ik_time 3
|           |   ...
```
</details>
<br>

You can visualize the comparison between IK result and normalized data:
```
python vis_demo.py --instrument cello --proj_first 1 --proj_last 85
```

<img src="assets/ik_fit_result.gif" alt="ik fit result" width="43%" height="43%" />


### 2.3 Data Segmentation
Run `python data_segmentation.py` to get the processed train data and the SPD-GEN dataset.

<details>
<summary>Processed Data Directory</summary>

```
ELGAR
└───data_process
|   |   ...
│   │   data_normalization.py
│   └───train_data
│       └───wholebody_normalized
|       └───wholebody_processed
|           |   audio.npy
|           |   motion.hdf5
```

</details>


## 2. Training
Run `bash train_script.sh` to train the model.



## 3. Validation

### 3.1. Qualitative Visualization

Our visualization is implemented via open3d.

`view_select = True` for the view select based on the monitor (resolution) you are using.

`view_select = False` for the video generation based on the selected view.

```
pip install open3d==0.14.1
python visualize.py
```

### 3.1. Quantitative Evaluation
Download the [test split](https://forms.gle/pF7KMZ2wrnFQCbY87) of SPD-GEN to quantitatively validate our results. Place them in the following directory.

<details>
<summary>Test Data Directory</summary>

```
ELGAR
|   ...
└───dataset
│   └───SPD-GEN
│       └───test_data
│           └───cello
|               └───audio
|               └───motion(optional, but recommend)
```
</details>

Run the following code to get the test motion and the test audio feature.
```
cd <your_path>/ELGAR/data_process
python prepare_inference_data.py --data_type test
```

Run `bash train_script.sh` and `bash test_script.sh` to train with train set and generate the motion with the test set audio (Uncomment the validation part).

Finally, run `python eval_testset.py` to get the evaluation results.

## 4. Retargeting

We retarget the motion from SMPL-X to [MetaHuman]("https://www.metahuman.com/") within the [Unreal Engine (UE)]("https://www.unrealengine.com/") to facilitate more applications. Despite the considerable challenges posed by the intricate and interactive nature of the performance motions—leading to suboptimal retargeting results—this work aims to motivate future research to advance more effective solutions.

We utilized the [SMPL-X Blender Add-on]("https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon") to export SMPL-X–based motion sequences into the FBX format to ensure compatibility with UE. Please follow the guide on the official website to install the add-on.

To prepare files for the add-on, run the following lines.
```
cd <your_path>/ELGAR/retargeting
python get_blender_animation_from_gen.py    # Conversion of the generated results
python get_blender_animation_from_train.py  # Conversion of the dataset samples
```

[Generate the fbx](./retargeting/README.md) by the add-on.

[Download](https://forms.gle/pF7KMZ2wrnFQCbY87) the UE (5.5.4) project and replace the animation with yours. Have Fun!

<img src="assets/retargeting.gif" alt="ik fit result" width="68%" height="68%" />

## Acknowledgements

Our code is built on the shoulders of giants. We would like to thank the following open-source projects: [MDM](https://github.com/GuyTevet/motion-diffusion-model), [EDGE](https://github.com/Stanford-TML/EDGE), [VPoser](https://github.com/nghorbani/human_body_prior), [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib), [Jukebox](https://github.com/openai/jukebox).

## License

This project is licensed under a custom Non-Commercial License.  
See the [LICENSE](./LICENSE) file for details.  
Commercial use is prohibited without permission.
