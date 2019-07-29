---


---

<h1 id="pseudo-lidar-on-argoverse-dataset-user-manual">Pseudo-Lidar on Argoverse Dataset User Manual</h1>
<p>This manual details the steps on how to reproduce our work of applying Pseudo-Lidar algorithm to the Argoverse Dataset. This work is dedicated on SJSU HPC platform. which integrate a GPU.</p>
<h2 id="get-the-pseudo-lidar-repository">1. Get the Pseudo-Lidar repository</h2>
<p><code>git clone https://github.com/MengshiLi/pseudo_lidar.git</code><br>
Assume your git folder is $ROOT for the following reference.</p>
<h2 id="download-the-argoverse-dataset">2. Download the Argoverse Dataset</h2>
<ul>
<li>Download link to dataset: <a href="https://www.argoverse.org/data.html#download-link">Argoverse Data Set Download</a></li>
<li>Unzip files to the same position, requires 250G+ storage capacity.</li>
<li>Data path will be represented as <code>$DATAPATH</code>.</li>
<li>For example, on our HPC: <code>$DATAPATH= /data/cmpe297-03-sp19/PilotA/Argoverse_3d_tracking/argoverse-tracking</code></li>
</ul>
<h2 id="setup-python-virtualenv-for-argoverse">3. Setup Python virtualenv for Argoverse</h2>
<ul>
<li>Clone the argoverse api: <code>git clone https://github.com/MengshiLi/argoverse-api</code></li>
<li>Login to HPC: <code>ssh &lt;SID&gt;@coe-hpc.sjsu.edu</code></li>
<li>Install virtual environment from this footholder node:</li>
</ul>
<pre><code>	module load python3/3.7.0
	which python3 #to obtain &lt;absolute-python-path&gt; for python3
	virtualenv --python=&lt;absolute-python-path&gt; venv-3.7-argo
	source ~/venv-3.7-argo/bin/activate
	pip install -e &lt;path_to_root_directory_of_the_repo&gt;
</code></pre>
<ul>
<li>The work load will be run on the GPU node, but GPU node has no network access, therefore, the env must be setup from the entrance node.</li>
</ul>
<h2 id="setup-python-virtualenv-for-psmnet">4. Setup Python virtualenv for PSMNet</h2>
<ul>
<li>PSMNet requires python2 and pytorch, therefore, it is better to setup a separate virtualenv.</li>
<li>Install from the footholder:</li>
</ul>
<pre><code>	module load cuda/9.2 python2/2.7.15
	which python #to obtain &lt;absolute-python-path&gt; for python2
	virtualenv --python=&lt;absolute-python-path&gt; venv_PSMNet_cuda
	source ~/venv_PSMNet_cuda/bin/activate
	pip install torch torchvision scikit-image
</code></pre>
<ul>
<li>Run from GPU node.</li>
</ul>
<h2 id="generate-groundtruth-disparity-from-lidar">5. Generate groundtruth disparity from Lidar</h2>
<p>The groundtruth disparity is used to finetuen the PSMNet model. Source code for generate disparity: <code>$ROOT/preprocessing/rgoisp.py</code>. Run it from any GPU node. Avoid running any heavy load on the footholder node.<br>
Before running the code below, ensure Argoverse dataset is organized in the following format.</p>
<pre><code>argoverse-tracking/
	train1/
	    log_id11/ # unique log identifier  
	        lidar/ # lidar point cloud file in .PC  
	        stereo_front_left/ # stereo left image
	        stereo_front_right/ # stereo right image
	        vehicle_calibration_info.json
	    log_id12/
		    ...
	train2/ 
		...
	train4/  
</code></pre>
<ul>
<li>Use screen to avoid task interruption due to shell stall: <code>screen</code></li>
<li>Obtain the GPU node: <code>srun -p gpu --gres=gpu --pty /bin/bash</code></li>
<li>From GPU node, re-activate the Python virtualenv for Argoverse:</li>
</ul>
<pre><code>	module load python3/3.7.0
	source ~/venv-3.7-argo/bin/activate
	python $ROOT/preprocessing/argo_gen_disp.py
</code></pre>
<h2 id="train-finetune-the-psmnet-model">6. Train: finetune the PSMNet model</h2>
<p>Download the pretrained model: <a href="https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing">PSMNet on KITTI2015</a>.<br>
Related source code:</p>
<ul>
<li><code>$ROOT/psmnet/dataloader/ARGOLoader3D.py</code></li>
<li><code>$ROOT/psmnet/dataloader/ARGOLoader_dataset3d.py</code></li>
<li><code>$ROOT/psmnet/finetune_3d_argo.py</code></li>
</ul>
<p>Launch a GPU node: <code>srun -p gpu --gres=gpu --pty /bin/bash</code><br>
From GPU:</p>
<pre><code>	module load cuda/9.2 python2/2.7.15
	source ~/venv_PSMNet_cuda/bin/activate
	python $ROOT/psmnet/finetune_3d_argo.py --loadmodel &lt;path to pretrained model&gt;
</code></pre>
<h2 id="inference-predict-disparity-from-stereo-image">7. Inference: predict disparity from stereo image</h2>
<p>Still running on the above virtualenv:</p>
<pre><code>python $ROOT/psmnet/argo_inference.py --datapath $DATAPATH --sub_folder train4 --loadmodel &lt;finetuned model path&gt;
</code></pre>
<h2 id="generate-pseudo-lidar-from-predicted-disparity">8. Generate Pseudo-Lidar from predicted disparity</h2>
<ul>
<li>
<p>Obtain another GPU node: <code>srun -p gpu --gres=gpu --pty /bin/bash</code></p>
</li>
<li>
<p>From GPU node, re-activate the python virtual environment for argoverse:</p>
</li>
</ul>
<pre><code>	module load python3/3.7.0
	source ~/venv-3.7-argo/bin/activate
</code></pre>
<ul>
<li>Usage:</li>
</ul>
<pre><code>python $ROOT/preprocessing/rgoGgen_lidar.py --datapath $DATAPATH --sub_folder train4
</code></pre>
<h2 id="apply-pseudo-lidar-to-any-3d-object-detection-model">9. Apply Pseudo-Lidar to any 3D object-detection model</h2>

