# 运行说明
大学三年级上册做的阿里云天池街景
做初步说明，目前有反馈说无法直接运行，据此在这说明一下，在tianchi文件夹内的images文件夹内本应有图片，上传受限，试过很多次都不行，分别对应着外面那几个mchar_test,mchar_train,mchar_val，把它复制粘贴到这images里面三个文件夹就好了
数据集分配问题解决了之后我们要修改coco.yaml,models文件夹下面yolov5s.yaml文件的东西，
coco.yaml里面存放的是train.txt,和val.txt的存放路径，以及类别数量nc，和类别名字name要改成你个标签对应地一个字典：也就是我们刚刚代码生成的文件路径， yolov5s.yaml改一下类别就可以了，改成你自己的类别数量。基本上没有需要改的了，我们直接打开train.py看看：直接划到最下面参数部分，代码里面的data，从data里面的coco.yaml读取train.txt（训练集的地址），train.txt就是存放的图片路径这样就可以读取到图片，标签的话自动定位到对应到labels文件的标签，weights就是预训练模型，epoch训练的轮数，我一般设置的是200，300，batch_size是多少张图片打包，如果电脑不好就是1，2，8，16，32，64，32应该是效果最佳，device:设置为0，workers:代表启动的线程数，如果电脑不好就设置为0，表示主线程。
在tianchi文件夹下那几个yaml都是这个街景我用过的，就是在原本coco.yaml基础上改到的。
2.配置训练文件
（1）准备功能
有了环境以及YOLOv5的源码，就可以开始训练模型啦！首先将YOLOv5项目文件夹与数据集文件夹放在同一个目录下，如下图所示。
（2）配置数据集yaml文件\n在yolov5项目的data文件夹下找到coco128.yaml，对其复制一份改名为dataset.yaml在data文件夹下。将里面的内容改成自己数据集的目录和标签名,多余的东西都删掉。(数据集目录可以直接和我的一样，下面的names需要根据你们自己的标签类别和个数修改)
（3）配置模型训练yaml文件\n在yolov5项目的models文件夹下找到yolov5s.yaml，对其复制一份在tianchi文件夹下并改名为street_yolov5s.yaml。然后将里面的nc改成自己数据集的分类个数,我的数据集有十个分类所以我填的是10。
3.模型训练\n现在就可以开始训练了，训练是通过yolov5的train.py进行，点开yolov5项目的train.py，下滑移动至433行的parse_opt函数，训练前需要修改一些参数，需要修改的参数如下表所示：
修改说明
436行的default参数改成模型训练配置文件的路径'--cfg'这行的default
437行的default参数改成数据集配置文件的路径'--data'这行的default
439行default参数改成训练的轮次，这里默认100，训练的轮次'--epochs'
440行default参数改成 batch 的大小，这里设置16，指定每个 batch 的大小，根据你自己电脑的配置设置，性能越好可以设置越高，性能不好的电脑就设置低一点，不然会报错跑不起'--batch-size'
440行default参数改成加载数据的进程数，默认为8，这里设置2\t加载数据的进程数，例如 --workers 8 表示使用 8 个进程来加载数据，也和电脑性能有关，一般电脑都设置不了多高，设个2就够了，如果发现2还是会报错跑不起训练就改成0
修改完上述参数之后就可以直接运行train.py，开始训练模型啦，最后生成的模型以及相关文件夹会保存在runs/train/exp文件夹下!!!
4.常见报错解决
（1）【训练没报错但是一直卡着不动】\n解决方法： 440行wokers的default参数改成0
（2）【RuntimeError: CUDA out of memory】\n这个错误表明在使用 PyTorch 运行时，CUDA 显存不足，导致无法分配所需的内存。这时可以通过减少批处理大小（Batch Size）来试试解决。\n解决方法： 修改train.py440行的batch-size的default参数再小一点。
（3）【export GIT_PYTHON_REFRESH=quiet】\n是一个用于设置环境变量的命令，它告诉 GitPython 库在加载之前不要刷新 Git 仓库的状态。在 Python 中，可以使用 os 模块来设置环境变量将GIT_PYTHON_REFRESH 的值设为 quiet。
解决方法： 在train.py的最开头import导库的地方，在import os下面加上以下代码即可。
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
三、利用模型进行目标检测
训练完模型之后，就可以通过训练好的模型进行目标检测了。
跑检测是通过yolov5的detect.py进行，点开yolov5项目的detect.py，下滑移动至218行的parse_opt函数，跑检测前需要修改一些参数，需要修改的参数如下表所示：
修改说明
220行（parse_opt函数内的第二行）的default参数改成你训练好的best.pt模型的路径，这里填写你模型的路径，best.pt会在你训练的生成在runs/train/exp文件中的weights文件夹中，weights文件夹中会有两个pt模型，best.pt代表所有训练轮次中最优的那个，last.pt则代表最后一轮训练的pt，所以一般我们都是选择best.pt
221行（parse_opt函数内的第三行）的default参数改成你需要跑检测的图像或视频等的路径\t这里填写你需要被检测的目标路径，图个是单张图片或者单个视频就填你单张图像或者单个视频的路径，如果是有很多张图和视频，你可以把这些图和视频放在一个文件夹内，然后这里填写你这个文件夹路径就可以对文件夹内所有的图像或者视频进行检测了
224行（parse_opt函数内的第六行）default参数默认0.25，这个代表置信度阈值，只有你的目标被检测的置信度大于你设置的这个值的时候，才会被当作目标然后用矩形框框出\n设置完上面的参数之后就可以直接运行detect.py跑检测啦，最终的检测结果会保存在runs/detect/exp文件夹下！

参考自http://t.csdnimg.cn/F23Um
