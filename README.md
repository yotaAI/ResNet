
# Resnet 

🛞 ResNet enables the construction of deeper neural networks, with more than a hundred layers, which was previously impossible due to the vanishing gradient problem.

🛞 I [@yota](https://github.com/yotaAI) am Implementing the model from Paper 📄 : https://arxiv.org/pdf/1512.03385


## 📝 Knowledge

The model is having seperate `Residual` connection block, for this with increasing number of layers the accuracy of the model will also increase. 


✏️ I am training the model with Imagenet Dataset.

✏️ For Learning Purpose we are initializing the model with random weights.

✏️ I am using one scheduler to devide the learning rate by `10` when the test loss come to a stable position.
 

## 🗃️ Dataset

🗞️ Training on Imagenet Dataset with 1000 classes. For dataset [Click](https://www.image-net.org/challenges/LSVRC/index.php).

🗞️As described in paper[Page 4] The model is first trained on dataset of `Scale=256`  then the pretrained model is again trained on `Scale=384` with `Learning Rate = 10^-3`.


## 🤖 Training VGG

🏷️ Now run the `train.py` and Boom!🤯
```bash
python3 train.py
```

🏷️ Model will be saved in `./resnet/`.




## 🥷🏻 Ninja Tech

⚡︎ Machine Learning ⚡︎ Deep Learning ⚡︎ CNN ⚡︎


