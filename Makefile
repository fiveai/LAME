# ------ GPU option ---------

GPUS=all# use e.g. GPUS=0,1,2 if you only want to use a certain set of GPUs on the cluster
GPU_FREE_UNDER=20# Usgae of memory (in MB) under which a GPU is considered free for simulation.


# ------ SERVER options --------- (only useful if you want to deploy/import results from remote server)

USER=# set the user of the remote server
SERVER_IP=# set the ip address of the remote server
SERVER_PATH=# set the path where you want all the folders to be dropped

# ------ Simulation mode params ------------

DEBUG=False
OVERRIDE=False
MODE=benchmark
SAVE_PLOTS=False

# ------ Simulation hyperparams ------------

DATASET=all
LOOP_ARG=MODEL.DEVICE
LOOP_VALUES=cuda
MAX_SIZE_PER_EPISODE=5e4
.DEFAULT_GOAL := help
.PHONY: help

# ------ Default options ------------

METHODS=non_adaptive tent pl shot lame ada_bn# Which methods to iterate benchmarking over
PROVIDER=msra
DEPTH=50
model=$(PROVIDER)_nft_r$(DEPTH)# Model used. For exhaustive list, c.f. configs/model/adaptation/
method=non_adaptive# Current method used. For exhaustive list, c.f. configs/model/adaptation/
data=iid_balanced# Current data mode used. For exhaustive list, c.f. configs/data/adaptation/

model_cfg=configs/model/$(model).yaml
method_cfg=configs/method/default/$(method).yaml
data_cfg=configs/data/$(data).yaml

# ---------- Plot options ------------

LABELS=ADAPTATION.METHOD
LATEX=False


# ------------------------- Data ----------------------------
# -----------------------------------------------------------

data/tao:
	python3 -m src.data.datasets.tao

data/imagenet_vid:
	python3 -m src.data.datasets.imagenet_vid

data/imagenet_c:
	python3 -m src.data.datasets.imagenet_c

data/imagenet_v2:
	python3 -m src.data.datasets.imagenet_v2

# ------------ Archiving results ----------------
# -----------------------------------------------

restore: # Restore experiments to output/
	python src/utils/list_files.py archive/$(MODE) output tmp.txt ; \
	read -r out_files < tmp.txt ; \
	mkdir -p output/$(MODE)/$${folder[1]} ; \
	for file in $${out_files}; do \
		cp -Rv $${file} output/$(MODE)/$${folder[1]}/ ; \
	done
	rm tmp.txt

store: # Archive experiments from output/ to archive/
	python src/utils/list_files.py output/$(MODE) archive tmp.txt
	{ read -r out_files; read -r archive_dir; } < tmp.txt ; \
	for file in $${out_files}; do \
		cp -Rv $${file} $${archive_dir}/ ; \
	done
	rm tmp.txt

# --------------- Fig 1 in paper ---------------
# ----------------------------------------------

nam_failure: checkpoints/msra/R-50.pkl
	make MODE=test DATASET=imagenet_val method=non_adaptive data=niid_balanced run
	make MODE=test DATASET=imagenet_val method=tent data=niid_balanced LOOP_ARG=ADAPTATION.LR LOOP_VALUES="0.001 0.01 0.1" run

plot_nam:
	make LABELS=ADAPTATION.LR DATASET=imagenet_val plot_metrics

# --------------------------- Validation ---------------------------
# ------------------------------------------------------------------

validation: checkpoints/msra/R-50.pkl
	all_datas="iid_balanced iid_imbalanced niid_balanced niid_imbalanced" ;\
	for method in $(METHODS); do \
		for data in $${all_datas}; do \
 			make MODE=validation DATASET=imagenet_c_16 method=$${method} data=$${data} run ;\
 			make MODE=validation DATASET=imagenet_c_val method=$${method} data=$${data} run ;\
 			make MODE=validation DATASET=imagenet_val method=$${method} data=$${data} run ;\
		done ;\
	done ;\

validation_heatmap:
	python3 -m src.utils.read_results \
	--stage validation \
	--latex $(LATEX) \
	--action cross_cases \
	--methods "NonAdaptiveMethod" "Tent" "AdaBN" "Shot" "PseudoLabeller" "LAME" \
	--datasets imagenet_val imagenet_c_val  imagenet_c_16 \
	--cases  \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=False"  \
		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=False"  \
	--case_names "IN" "IN + niid"  "IN + ls" "IN + ls + niid" \
				 "INC" "INC + niid"  "INC + ls" "INC + ls + niid" \
				 "INC_16" "INC_16 + niid" "INC_16 + ls" "INC_16 + ls + niid" 

				 
save_best_config:
	python3 -m src.utils.read_results \
	--stage validation \
	--action log_best \
	--latex $(LATEX) \
	--save \
	--save_name overall_best \
	--datasets imagenet_val imagenet_c_val  imagenet_c_16 \
	--cases  \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_val'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=False"  \
		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False"  \
   		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True" \
   		   "DATASETS.ADAPTATION=['imagenet_c_16'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=False"  \
	--case_names "IN" "IN + niid"  "IN + ls" "IN + ls + niid" \
				 "INC" "INC + niid"  "INC + ls" "INC + ls + niid" \
				 "INC_16" "INC_16 + niid" "INC_16 + ls" "INC_16 + ls + niid" 


# ----------------------- Testing --------------------
# ----------------------------------------------------

test: checkpoints/msra/R-50.pkl
	datas="iid_balanced iid_imbalanced" ;\
	for data in $${datas}; do \
		for method in $(METHODS); do \
			make MODE=benchmark DATASET=imagenet_v2 method=$${method} data=$${data} run ;\
			make MODE=benchmark DATASET=imagenet_c_test method=$${method} data=$${data} run ;\
		done ;\
	done ;\
	datas="niid_balanced" ;\
	for data in $${datas}; do \
		for method in $(METHODS); do \
			make MODE=benchmark DATASET=imagenet_v2 method=$${method} data=$${data} run ;\
			make MODE=benchmark DATASET=imagenet_vid_val method=$${method} data=$${data} run ;\
			make MODE=benchmark DATASET=tao_trainval method=$${method} data=$${data} run ;\
		done ;\
	done ;\

plot_box:
	python3 -m src.utils.read_results \
	--stage benchmark \
	--latex $(LATEX) \
	--save_name iid_balanced_$(DEPTH) \
	--methods "NonAdaptiveMethod" "Tent" "Shot" "LAME" "AdaBN" "PseudoLabeller" \
	--action benchmark_box \
	--title "(a) I.I.D with Posterior Shift" \
	--cases  \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),DATASETS.ADAPTATION=['imagenet_v2'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),DATASETS.ADAPTATION=['imagenet_c_test'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=True" \
	--case_names "IV2-IID/B" "IC-IID/B"

	python3 -m src.utils.read_results \
	--stage benchmark \
	--latex $(LATEX) \
	--save_name iid_imbalanced_$(DEPTH) \
	--title "(b) I.I.D with Posterior Shift + Prior Shift" \
	--methods "NonAdaptiveMethod" "Tent" "Shot" "LAME" "AdaBN" "PseudoLabeller" \
	--action benchmark_box \
	--cases  \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),DATASETS.ADAPTATION=['imagenet_v2'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True" \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),DATASETS.ADAPTATION=['imagenet_c_test'],DATASETS.IMBALANCE_SHIFT=True,DATASETS.IID=True" \
	--case_names "IV2-IID/I" "IC-IID/I"

	python3 -m src.utils.read_results \
	--stage benchmark \
	--latex $(LATEX) \
	--save_name niid_$(DEPTH) \
	--methods "NonAdaptiveMethod" "Tent" "Shot" "LAME" "AdaBN" "PseudoLabeller" \
	--title "(c) N.I.I.D with Posterior Shift + Prior Shift" \
	--action benchmark_box \
	--cases  \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),DATASETS.ADAPTATION=['imagenet_v2'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False" \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),DATASETS.ADAPTATION=['imagenet_vid_val'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False" \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),DATASETS.ADAPTATION=['tao_trainval'],DATASETS.IMBALANCE_SHIFT=False,DATASETS.IID=False" \
	--case_names "IV2-NIID/B" "IVid-NIID" "Tao-NIID"

# --------------------- Study of batch size -------------------------------
# -------------------------------------------------------------------------

study_batch_size: checkpoints/msra/R-50.pkl
	make LOOP_ARG=ADAPTATION.BATCH_SIZE LOOP_VALUES="128" test

plot_batch:
	python3 -m src.utils.read_results \
	--stage benchmark \
	--latex $(LATEX) \
	--save_name $(DEPTH) \
	--action benchmark_batch \
	--methods "NonAdaptiveMethod" "LAME" "AdaBN" "Tent" "PseudoLabeller" "Shot" \
	--title "Test" \
	--cases  \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),ADAPTATION.BATCH_SIZE=16" \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),ADAPTATION.BATCH_SIZE=32" \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),ADAPTATION.BATCH_SIZE=64" \
		   "MODEL.RESNETS.DEPTH=$(DEPTH),ADAPTATION.BATCH_SIZE=128" \
	--case_names "16" "32" "64" "128"


# --------------- Robustness w.r.t training procedure -------------------
# -----------------------------------------------------------------------

robustness_training: checkpoints/pytorch/R-50.pth checkpoints/pytorch/R-50.pth checkpoints/simclr/R-50.pth
	for provider in simclr msra pytorch; do \
		make PROVIDER=$${provider} test ;\
	done ;\

plot_spider_training:
	python3 -m src.utils.read_results \
	--stage benchmark \
	--latex $(LATEX) \
	--save_name different_r50_training \
	--action benchmark_spider \
	--cases  \
		   "MODEL.WEIGHTS='checkpoints/msra/R-50.pkl'" \
		   "MODEL.WEIGHTS='checkpoints/simclr/R-50.pth'" \
		   "MODEL.WEIGHTS='checkpoints/pytorch/R-50.pth'" \
	--case_names "MSRA_R50" "SIMCLR_R50" "PYTORCH_R50"

# --------------- Robustness w.r.t architecture -------------------------
# -----------------------------------------------------------------------

robustness_arch: checkpoints/pytorch/R-18.pth checkpoints/msra/R-50.pkl checkpoints/msra/R-101.pkl checkpoints/vit/B-16.pth checkpoints/pytorch/EN-b4.pth
	for model in pytorch_nft_r18 msra_nft_r50 msra_nft_r101 pytorch_nft_eb4 vit_nft_b16; do \
		make model=$${model} test ;\
	done ;\

plot_spider_arch:
	python3 -m src.utils.read_results \
	--stage benchmark \
	--latex $(LATEX) \
	--save_name different_arch \
 	--methods "NonAdaptiveMethod" "LAME" "Shot" "PseudoLabeller" "Tent" \
	--action benchmark_spider \
	--cases  \
		   "MODEL.WEIGHTS='checkpoints/pytorch/R-18.pth'" \
		   "MODEL.WEIGHTS='checkpoints/msra/R-50.pkl'" \
		   "MODEL.WEIGHTS='checkpoints/msra/R-101.pkl'" \
		   "MODEL.WEIGHTS='checkpoints/vit/B-16.pth'" \
		   "MODEL.WEIGHTS='checkpoints/pytorch/EN-b4.pth'" \
	--case_names "RN-18" "RN-50" "RN-101" "ViT-B" "EN-B4"

# --------------- Study of runtimes -------------------
# -----------------------------------------------------

runtimes: checkpoints/pytorch/R-18.pth checkpoints/pytorch/R-50.pth checkpoints/pytorch/R-101.pth checkpoints/vit/B-16.pth checkpoints/pytorch/EN-b4.pth
	methods="tent lame" ;\
	provider="pytorch" ;\
	for method in $${METHODS}; do \
		for depth in 18 50 101; do \
			make MAX_SIZE_PER_EPISODE=1e4 DATASET=imagenet_val PROVIDER=$${provider} DEPTH=$${depth} method=$${method} data=niid_balanced LOOP_ARG=ADAPTATION.BATCH_SIZE LOOP_VALUES="64" run ;\
		done ;\
		make MAX_SIZE_PER_EPISODE=1e4 DATASET=imagenet_val model=pytorch_nft_eb4 method=$${method} data=niid_balanced LOOP_ARG=ADAPTATION.BATCH_SIZE LOOP_VALUES="16" run ;\
		make MAX_SIZE_PER_EPISODE=1e4 DATASET=imagenet_val model=vit_nft_b16 method=$${method} data=niid_balanced LOOP_ARG=ADAPTATION.BATCH_SIZE LOOP_VALUES="16" run ;\
	done ;\


plot_time:
	make DATASET=imagenet_val plot_metrics


# --------------- Study of affinity matrix -------------------
# -----------------------------------------------------

affinity_robustness:
	make METHODS=non_adaptive test
	make METHODS=lame LOOP_ARG=ADAPTATION.LAME_AFFINITY LOOP_VALUES="kNN linear rbf" robustness_arch

affinity_plot:
	python3 -m src.utils.read_results \
	--stage benchmark \
	--latex $(LATEX) \
	--save_name different_arch \
 	--methods "LAME" \
 	--method_params ADAPTATION.LAME_AFFINITY \
	--action benchmark_spider \
	--cases  \
		   "MODEL.WEIGHTS='checkpoints/pytorch/R-18.pth'" \
		   "MODEL.WEIGHTS='checkpoints/msra/R-50.pkl'" \
		   "MODEL.WEIGHTS='checkpoints/msra/R-101.pkl'" \
		   "MODEL.WEIGHTS='checkpoints/vit/B-16.pth'" \
		   "MODEL.WEIGHTS='checkpoints/pytorch/EN-b4.pth'" \
	--case_names "RN-18" "RN-50" "RN-101" "ViT-B" "EN-B4"

# ---------------- Download models / convert them ----------
# ----------------------------------------------------------

# SIMClr models come from https://github.com/google-research/simclr

checkpoints/msra/R-50.pkl:
	mkdir -p checkpoints/msra
	wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl -O checkpoints/msra/R-50.pkl

checkpoints/msra/R-101.pkl:
	mkdir -p checkpoints/msra
	wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl -O checkpoints/msra/R-101.pkl

checkpoints/simclr/R-50.pth:
	mkdir -p checkpoints/simclr/unconverted
	gsutil -m cp -r "gs://simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0/" checkpoints/simclr/unconverted/
	python src/modeling/convert_simclr_models.py --input checkpoints/simclr/unconverted/r50_1x_sk0 --out checkpoints/simclr/R-50.pth ;
	unzip checkpoints/simclr/unconverted/r50_1x_sk0.zip -d checkpoints/simclr/unconverted/

checkpoints/simclr/R-101.pth:
	mkdir -p checkpoints/simclr/unconverted
	gsutil -m cp -r "gs://simclr-checkpoints/simclrv2/pretrained/r101_1x_sk0" checkpoints/simclr/unconverted/
	python src/modeling/convert_simclr_models.py --input checkpoints/simclr/unconverted/r101_1x_sk0 --out checkpoints/simclr/R-101.pth ;


checkpoints/pytorch/unconverted/R-18.pth:
	mkdir -p checkpoints/pytorch/unconverted
	wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O checkpoints/pytorch/unconverted/R-18.pth

checkpoints/pytorch/unconverted/R-50.pth:
	mkdir -p checkpoints/pytorch/unconverted
	wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O checkpoints/pytorch/unconverted/R-50.pth

checkpoints/pytorch/unconverted/R-101.pth:
	mkdir -p checkpoints/pytorch/unconverted
	wget https://download.pytorch.org/models/resnet101-63fe2227.pth -O checkpoints/pytorch/unconverted/R-101.pth

checkpoints/vit/unconverted/B-16.pth:
	mkdir -p checkpoints/vit/unconverted
	wget https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth -O checkpoints/vit/unconverted/B-16.pth

checkpoints/vit/unconverted/L-16.pth:
	mkdir -p checkpoints/vit/unconverted
	wget https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth -O checkpoints/vit/unconverted/L-16.pth

checkpoints/pytorch/unconverted/EN-b4.pth:
	mkdir -p checkpoints/pytorch/unconverted
	wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth -O checkpoints/pytorch/unconverted/EN-b4.pth

checkpoints/pytorch/unconverted/EN-b7.pth:
	mkdir -p checkpoints/pytorch/unconverted
	wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth -O checkpoints/pytorch/unconverted/EN-b7.pth


checkpoints/pytorch/R-18.pth: checkpoints/pytorch/unconverted/R-18.pth src/modeling/convert_pytorch_models.py
	python src/modeling/convert_pytorch_models.py --input checkpoints/pytorch/unconverted/R-18.pth --out checkpoints/pytorch/R-18.pth

checkpoints/pytorch/R-50.pth: checkpoints/pytorch/unconverted/R-50.pth src/modeling/convert_pytorch_models.py
	python src/modeling/convert_pytorch_models.py --input checkpoints/pytorch/unconverted/R-50.pth --out checkpoints/pytorch/R-50.pth

checkpoints/pytorch/R-101.pth: checkpoints/pytorch/unconverted/R-101.pth src/modeling/convert_pytorch_models.py
	python src/modeling/convert_pytorch_models.py --input checkpoints/pytorch/unconverted/R-101.pth --out checkpoints/pytorch/R-101.pth

checkpoints/vit/B-16.pth: checkpoints/vit/unconverted/B-16.pth src/modeling/convert_vit_models.py
	python src/modeling/convert_vit_models.py --input checkpoints/vit/unconverted/B-16.pth --out checkpoints/vit/B-16.pth

checkpoints/vit/L-16.pth: checkpoints/vit/unconverted/L-16.pth src/modeling/convert_vit_models.py
	python src/modeling/convert_vit_models.py --input checkpoints/vit/unconverted/L-16.pth --out checkpoints/vit/L-16.pth

checkpoints/pytorch/EN-b4.pth: checkpoints/pytorch/unconverted/EN-b4.pth src/modeling/convert_efficient_net.py
	python src/modeling/convert_efficient_net.py --input checkpoints/pytorch/unconverted/EN-b4.pth --out checkpoints/pytorch/EN-b4.pth

checkpoints/pytorch/EN-b7.pth: checkpoints/pytorch/unconverted/EN-b7.pth src/modeling/convert_efficient_net.py
	python src/modeling/convert_efficient_net.py --input checkpoints/pytorch/unconverted/EN-b7.pth --out checkpoints/pytorch/EN-b7.pth

# ----------------- Miscellaneous ----------------------------
# ------------------------------------------------------------

kill_all: ## Kill all my python and tee processes on the server
	ps -u $(USER) | grep "python" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill
	ps -u $(USER) | grep "tee" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill

delete_cache:
	find . -type d -name '*__pycache__*' -exec rm -r {} \;


help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sed -n 's/^\(.*\): \(.*\)##\(.*\)/\1\3/p' \
	| column -t  -s ' '


run:
	export CUDA_DEVICE_ORDER=PCI_BUS_ID ;\
	for loop_val in $(LOOP_VALUES); do \
        ( \
            echo "==========================" ;\
            echo "Running $(data_cfg) $(model_cfg) $(method_cfg)..." ;\
            echo "==========================" ;\
            IFS='/.' read -r -a data_array <<< "$(data_cfg)" ;\
            IFS='/.' read -r -a method_array <<< "$(method_cfg)" ;\
            IFS='/.' read -r -a model_array <<< "$(model_cfg)" ;\
            OUTPUT=output/$(MODE)/$(DATASET)/$${data_array[-2]}_$${model_array[-2]}_$${method_array[-2]}_$(LOOP_ARG)=$${loop_val} ;\
            mkdir -p $${OUTPUT} ;\
            python -m src.main \
                --allowed_gpus $(GPUS) \
                --data-config $(data_cfg) \
                --method-config $(method_cfg) \
                --model-config $(model_cfg) \
                --mode $(MODE) \
                OUTPUT_DIR $${OUTPUT} \
                DATASETS.ADAPTATION "['$(DATASET)']" \
                OVERRIDE $(OVERRIDE) \
                DEBUG $(DEBUG) \
                SAVE_PLOTS $(SAVE_PLOTS) \
                DATASETS.MAX_SIZE_PER_EPISODE $(MAX_SIZE_PER_EPISODE) \
                $(LOOP_ARG) $${loop_val} \
                | tee $${OUTPUT}/raw_log.txt \
        ) & \
        sleep 10 ;\
	done \



plot_metrics:
	python3 -m src.utils.plot --stage $(MODE)  \
							  --latex $(LATEX)  \
							  --dataset $(DATASET) \
							  --labels $(LABELS) \
							  --folder 'numpy' \
							  --force_labels

# ------------ Transfer with servers -----------------
# ---------------------------------------------------

import/archive:
	rsync -av  --include="*/" --include "*.json" --include "*.png" --include "*accuracy*.npy" --include "*eigen*.npy" --include "*cond_ent*.npy" --include "*time*.npy" --include "*.yaml" --exclude "*" \
		$(SERVER_IP):$(SERVER_PATH)/archive/$(MODE)/ ./archive/$(MODE)

import/results:
	mkdir -p output/$(MODE)/
	rsync -av  --exclude "events.*" --exclude "*.pth" \
		$(SERVER_IP):$(SERVER_PATH)/output/$(MODE)/ ./output/$(MODE)/

import/metrics:
	rsync -av  --include="*/" --include "*.json" --include "*.yaml" --exclude "*" \
		$(SERVER_IP):$(SERVER_PATH)/output/ ./output/


import/plots:
	rsync -av  $(SERVER_IP):$(SERVER_PATH)/plots/ ./plots


import/tensorboard:
	rsync -av  --include="*/" --include "event.*" --exclude "*" \
		$(SERVER_IP):$(SERVER_PATH)/output/ ./output/

import/models:
	rsync -av  --include="*/" --include "model_final.pth" --exclude "*" \
		$(SERVER_IP):$(SERVER_PATH)/output/ ./output/

deploy:
	rsync -av  \
      --exclude .git \
      --exclude logs \
      --exclude archive \
      --exclude checkpoints \
      --exclude *.tar \
      --exclude training.log \
      --exclude results \
      --exclude __pycache__ \
      --exclude tmp \
      --exclude *.sublime-project \
      --exclude *.sublime-workspace \
      --exclude output \
      --exclude *.md \
      --exclude plots \
      --exclude lame \
      --exclude *.so \
      ./ $(SERVER_IP):$(SERVER_PATH)/
	rsync -av --delete  \
      ./configs/ $(SERVER_IP):$(SERVER_PATH)/configs/

deploy/msra:
	rsync -av  ./checkpoints/MSRA/ $(SERVER_IP):$(SERVER_PATH)/checkpoints/MSRA/

deploy/simclr:
	rsync -av  ./checkpoints/simclr/ $(SERVER_IP):$(SERVER_PATH)/checkpoints/simclr/
