# projetIDS

## 1. Détection et tracking
Tous les codes et la bonne architecture se trouvent dans le dossier /deep-learning.
#### Requirements:
+ CUDA 10.1
+ Python 3.8.5
+ Pytorch 1.7.1 - torchvision 0.8.2 
+ OpenCV 4.1.2.30 (contrib) 
+ autres packages principaux: cython, filterpy, numpy, numba, pandas, requests, scikit-learn (0.22.2), scipy, seaborn, tensorboard, yaml, wandb

Le script permettant de faire tourner la détection (YOLO v5) et le tracking (algorithme Sort) sur une vidéo est: `yolo_and_track.py`. Pour cela, il faut mettre le chemin de la vidéo à la ligne 38 `videopath`. La vidéo résultat RESULT.mp4 se trouvera dans /sort/results. 

Au cours du traitement de la vidéo par `yolo_and_sort.py`, des images de chaque individu vont être écrites dans le dossier PHOTOS (un sous-dossier par individu est créé, à l'intérieur duquel sont stockées les images de cet individu). 

Le réseau de neurones YOLO v5 utilisé (poids: `insect.pt`) est celui qui reconnaît des insectes: pour tester le code il vaut donc mieux utiliser une vidéo d'insectes (avec suffisamment de zoom). Sinon, il faut télécharger les poids `yolov5s.pt` (https://github.com/ultralytics/yolov5). 

Les scripts principaux appelés dans `yolo_and_track.py` sont: `detect_bis.py` et `send_data.py`.

## 2. Classification _fine grained_
Le code, à titre informatif, se trouve dans /deep-learning/RESNET_BUTTERFLIES: `infer_resnet_butterflies.py`. Il s'agit uniquement de l'inférence, qu'il faut faire tourner dans un terminal à part, pendant que `yolo_and_track` est lancé dans un premier terminal.

A chaque instant, le script vérifie si une nouvelle image a été écrite dans le dossier PHOTOS. Le cas échéant, la classification est effectuée et le résultat est stocké dans des fichiers `species.json` et `species.csv` (un exemple de chaque est disponible). Ici, les données des capteurs sont simulées aléatoirement.

