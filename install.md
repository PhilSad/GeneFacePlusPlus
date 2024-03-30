wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n geneface python=3.9
conda activate geneface
conda install -y conda-forge::ffmpeg # ffmpeg with libx264 codec to turn images to video

### We recommend torch2.0.1+cuda11.7. We found torch=2.1+cuda12.1 leads to erors in torch-ngp
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/facebookresearch/pytorch3d
pip install cython
pip install openmim==0.3.9
pip install chardet
apt install rsync
mim install mmcv==2.1.0 # use mim to speed up installation for mmcv

# other dependencies
apt update  && apt-get  install -y libasound2-dev portaudio19-dev
pip install -r docs/prepare_env/requirements.txt -v


# train

```bash
export VIDEO_ID=emily2
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./

#step0
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf "scale='if(lt(iw,ih),512,trunc(512*iw/ih/2)*2)':'if(lt(iw,ih),trunc(512*ih/iw/2)*2,512)',crop=512:512" data/raw/videos/${VIDEO_ID}_512.mp4
mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_512.mp4 data/raw/videos/${VIDEO_ID}.mp4


# step 1
mkdir -p data/processed/videos/${VIDEO_ID}
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 data/processed/videos/${VIDEO_ID}/aud.wav
python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID}

#step 2

mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background

#step 3
python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset  --debug --id_mode=global

#step4
python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}

bash data_gen/runs/nerf/run.sh ${VIDEO_ID}

# TRAIN

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/${VIDEO_ID} --reset

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/${VIDEO_ID}_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/${VIDEO_ID}_head --reset

CUDA_VISIBLE_DEVICES=0  python inference/genefacepp_infer.py --head_ckpt=checkpoints/motion2video_nerf/emily2 --torso_ckpt=checkpoints/motion2video_nerf/emily2_head150k_torso --drv_aud=/workspace/GeneFacePlusPlus/kelki.mp3







CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/emily2/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/emily2_head150k_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/emily2 --reset
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/macron/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/macron_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/macron --reset

```