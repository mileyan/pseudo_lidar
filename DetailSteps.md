---


---

<h1 id="pseudo-lidar-user-manual">Pseudo-Lidar User Manual</h1>
<h2 id="download-the-argoverse-dataset">Download the Argoverse Dataset</h2>
<ul>
<li>Download link to dataset: <a href="https://www.argoverse.org/data.html#download-link">Argoverse Data Set Download</a></li>
<li>Unzip files to the same position, requires 250G+ storage capacity.</li>
<li>Data path will be represented as <code>$DATAPATH</code>, on HPC: <code>$DATAPATH= /data/cmpe297-03-sp19/PilotA/Argoverse_3d_tracking/argoverse-tracking</code></li>
</ul>
<h2 id="get-the-pseudo-lidar-repository">Get the pseudo-lidar repository</h2>
<p><code>git clone https://github.com/MengshiLi/pseudo_lidar.git</code><br>
Navigate to the git folder, assume as $ROOT.</p>
<h2 id="setup-the-argoverse-environment-on-hpc">Setup the argoverse environment on HPC</h2>
<ul>
<li>Install virtual environment from the footholder node:</li>
</ul>
<pre><code>	ssh &lt;SID&gt;@coe-hpc.sjsu.edu
	git clone https://github.com/MengshiLi/argoverse-api
	module load python3/3.7.0
	which python3 #to obtain &lt;absolute-python-path&gt; for python3
	virtualenv --python=&lt;absolute-python-path&gt; venv-3.7-argo
	source ~/venv-3.7-argo/bin/activate
</code></pre>
<ul>
<li>From the activated virtualenv:<br>
<code>pip install -e &lt;path_to_root_directory_of_the_repo&gt;</code></li>
</ul>
<h2 id="generate-the-groundtruth-disparity-from-lidar">Generate the groundtruth disparity from lidar</h2>
<p>The groundtruth disparity is used to finetuen the PSMNet model. Run it from any GPU node. Avoid running any heavy load on the footholder node</p>
<ul>
<li>
<p>Use screen to avoid task interruption due to shell stall <code>screen</code></p>
</li>
<li>
<p>Obtain the GPU node: <code>srun -p gpu --gres=gpu --pty /bin/bash</code></p>
</li>
<li>
<p>From GPU node, re-activate the python virtual environment for argoverse:</p>
</li>
</ul>
<pre><code>	module load python3/3.7.0
	source ~/venv-3.7-argo/bin/activate
</code></pre>
<p>Source code for generate disparity:</p>
<pre><code>	$ROOT/preprocessing/ArgoGenDisp.py
</code></pre>
<p>Usage: <code>python $ROOT/preprocessing/ArgoGenDisp.py</code></p>
<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled=""> sean to update the latest source code and usage</li>
</ul>
<h2 id="setup-the-psmnet-virtual-environment">Setup the PSMNet virtual environment</h2>
<ul>
<li>Install from the footholder</li>
</ul>
<pre><code>ssh &lt;SID&gt;@coe-hpc.sjsu.edu
module load cuda/9.2 python2/2.7.15
which python #to obtain &lt;absolute-python-path&gt; for python2
virtualenv --python=&lt;absolute-python-path&gt; venv_PSMNet_cuda
source ~/venv_PSMNet_cuda/bin/activate
pip install torch torchvision scikit-image
</code></pre>
<h2 id="finetune-the-psmnet-model">Finetune the PSMNet model</h2>
<p>Related source code:</p>
<ul>
<li><code>$ROOT/psmnet/dataloader/ArgoLoader3D.py</code></li>
<li><code>$ROOT/psmnet/dataloader/Argoloader_dataset3d.py</code></li>
<li><code>$ROOT/psmnet/finetune_3d_argo.py</code></li>
</ul>
<p>Download the pretrained model: <a href="https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing">PSMNet on KITTI2015</a>.<br>
Launch a GPU node: <code>srun -p gpu --gres=gpu --pty /bin/bash</code><br>
From GPU:</p>
<pre><code>	module load cuda/9.2 python2/2.7.15
	source ~/venv_PSMNet_cuda/bin/activate
</code></pre>
<p>Usage:</p>
<pre><code>	python $ROOT/psmnet/finetune_3d_argo.py --loadmodel &lt;path to pretrained model&gt;
</code></pre>
<h2 id="use-the-finetuned-psmnet-to-predict-disparity-from-stereo-image">Use the finetuned PSMNet to predict disparity from stereo image</h2>
<p>Launch a GPU node: <code>srun -p gpu --gres=gpu --pty /bin/bash</code><br>
From GPU:</p>
<pre><code>	module load cuda/9.2 python2/2.7.15
	source ~/venv_PSMNet_cuda/bin/activate
</code></pre>
<p>Usage:</p>
<pre><code>	python $ROOT/psmnet/submission4_inference.py --datapath $DATAPATH --sub_folder train4 --loadmodel &lt;finetuned model path&gt;
</code></pre>
<h2 id="generate-pseudo-lidar-from-predicted-disparity">Generate pseudo-lidar from predicted disparity</h2>
<ul>
<li>
<p>Obtain the GPU node: <code>srun -p gpu --gres=gpu --pty /bin/bash</code></p>
</li>
<li>
<p>From GPU node, re-activate the python virtual environment for argoverse:</p>
</li>
</ul>
<pre><code>	module load python3/3.7.0
	source ~/venv-3.7-argo/bin/activate
</code></pre>
<p>Usage:</p>
<pre><code>	python $ROOT/preprocessing/ArgoGenLidar.py --datapath $DATAPATH --sub_folder train4
</code></pre>

