$ErrorActionPreference = "Stop"
$py = "py"
& $py -3.11 -m venv .venv311
& .\.venv311\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv311\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv311\Scripts\python.exe -m pip install --no-deps tensorflow-model-optimization==0.8.0
& .\.venv311\Scripts\python.exe -c "import tensorflow as tf, keras, tf_keras, tensorflow_model_optimization as tfmot; print('TensorFlow',tf.__version__); print('Keras',keras.__version__); print('tf_keras',tf_keras.__version__); print('TFMOT',tfmot.__version__)"
Write-Host "Environment ready. Activate with: .\.venv311\Scripts\Activate.ps1"
