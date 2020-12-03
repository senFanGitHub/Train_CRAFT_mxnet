## info
训练字符级图像分割

将带有字符级标注的数据准备成三元组（原图， 字符热力图，字符连接区域热力图）

输入原图，L2loss 训练输出预测的字符置信度热力图，字符连接区域置信度热力图。


## 环境
mxnet-cuxx

opencv

scipy


## 训练
jupyter step by step
or
python Train_CharDet.py


## 数据
目前开源的字符级标准数据有2个：ICDAR的ReCTs 和人工合成数据SynthText，该项目使用这两个数据训练

ICDAR的ReCTs：http://datasets.cvc.uab.es/rrc/ReCTS.zip

SynthText : http://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip,

https://github.com/ankush-me/SynthText


## 推理
参见core.py Train_Char_Det类的 infer方法。