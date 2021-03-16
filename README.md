# AI陪你看《动物世界》

### 还记得中央频道的《动物世界》么，从小看到大，再看最先想起来的还是赵忠祥老师的声音，童年啊~~
### 每一期都有新的主题，还是百看不厌，如今AI这么火，哪里能没有AI嘞！！！
### 五十行代码，带你走进Ai的《动物世界》

### 哔站：[哔哩哔哩](https://www.bilibili.com/video/BV1jU4y1a7eu)

### csdn：[csdn](https://blog.csdn.net/qq_38758774/article/details/114868301)

<iframe src="//player.bilibili.com/player.html?aid=672170288&bvid=BV1jU4y1a7eu&cid=311022390&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>


![image](https://wx4.sinaimg.cn/mw690/006AID3Hly1golgun0itfg30go09eb2h.gif)

### 安装模型


```python
!hub install mobilenet_v2_animals==1.0.0
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    2021-03-16 13:47:11,410 - INFO - Lock 139951854653264 acquired on /home/aistudio/.paddlehub/tmp/mobilenet_v2_animals
    [INFO 2021-03-16 13:47:11,410 filelock.py:274] Lock 139951854653264 acquired on /home/aistudio/.paddlehub/tmp/mobilenet_v2_animals
    Download https://bj.bcebos.com/paddlehub/paddlehub_dev/mobilenet_v2_animals_1.0.0.tar.gz
    [##################################################] 100.00%
    Decompress /home/aistudio/.paddlehub/tmp/tmpgvex3v__/mobilenet_v2_animals_1.0.0.tar.gz
    [##################################################] 100.00%
    [32m[2021-03-16 13:47:21,684] [    INFO][0m - Successfully installed mobilenet_v2_animals-1.0.0[0m
    2021-03-16 13:47:21,726 - INFO - Lock 139951854653264 released on /home/aistudio/.paddlehub/tmp/mobilenet_v2_animals
    [INFO 2021-03-16 13:47:21,726 filelock.py:318] Lock 139951854653264 released on /home/aistudio/.paddlehub/tmp/mobilenet_v2_animals
    [0m

#### **开始前有必要提醒下，导入的python包有些同学的环境可能没有安装**
#### **还是需要手动下载的，直接在cmd `pip install 包名 `即可这里不做演示了哈**

## 1.导入包
#### **这些都是常用的Opencv，PaddleHub，numpy，time还有画图和编辑视频用的  moviepy 和 PIL**


```python
import cv2
import paddlehub as hub
import numpy
import time
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm # to create font
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    2021-03-16 13:47:24,294 - INFO - font search path ['/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/afm', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts']
    [INFO 2021-03-16 13:47:24,294 font_manager.py:1071] font search path ['/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/afm', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts']
    2021-03-16 13:47:24,643 - INFO - generated new fontManager
    [INFO 2021-03-16 13:47:24,643 font_manager.py:1458] generated new fontManager


## 2.图片转类型
#### **因为我们接下来要将Ai识别的结果导入到 视频里面，所以我将视频内的每一帧都保存成图片，并转化成可以在图像上绘图的对象，添加上识别的结果，以便操作**


```python
# ---------------------------------转换图片------------------------------
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("/usr/share/fonts/truetype/arphic/ukai.ttc", textSize, encoding="utf-8")
        # "font/simsun.ttc", textSize, encoding="utf-8")     ##win环境使用
        # fm.findfont(fm.FontProperties(family='')), textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    
```

## 3. 读取视频帧并存储新的视频
#### **读取视频内每一帧，并且每5帧识别一次图片，将结果添加到帧上面，且一边读取一边写入输出的视频**


```python
# ------------------------------------读取视频帧--------------------------------- 
def open(path):
	cap = cv2.VideoCapture(path)  #打开原视频
	fps = cap.get(cv2.CAP_PROP_FPS)		#读取帧率
	print(fps)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # avi DIVX  mp4v
	videoWriter = cv2.VideoWriter('output.mp4',fourcc,fps,(1280,720))
								#输出视频地址，视频编码，帧率，画面大小（要和图片一致）
	print (cap.isOpened() )
	success = True
	i = 1
	while(success): 
		success, frame = cap.read() 
		if success==False:
			break
		cv2.imwrite("video.jpg" , frame) 
		if i%5==1 :		#每5帧判断一次
			i=2
			result = classifier.classification(images=[cv2.imread(r'video.jpg')])
			print(result)
			font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
		img = cv2ImgAddText(cv2.imread(r'video.jpg'), str(result[0]), 50, 130,  (255, 255, 255), 25)
										# 图像，       文字内容， 横坐标，纵坐标，  颜色，  字体大小
		videoWriter.write(img)        #把图片写进视频
		# cv2.imshow("animals",img)	
		if cv2.waitKey(100) & 0xff == ord('1'):#按下1退出
			break  
		i+=1
	cap.release()   #释放视频
	cv2.destroyAllWindows()  #删除窗口
	videoWriter.release() #释放编辑视频
```

## 4.运行脚本，识别视频内动物


```python
if __name__=='__main__':
	classifier = hub.Module(name="mobilenet_v2_animals")
	path=r"data/data74891/zong.mp4"  #  原视频路径
	video = VideoFileClip(path)
	audio = video.audio
	audio.write_audiofile('output.mp3')				#  提取原视频声音
	open(path)
```

### 运行脚本完成后，*.mp3和*.mp4就是我们AI识别后的音频和视频了
&nbsp; 
#### **问：为什么要提取mp3？**
#### **答：最后获取的视频时在原视频转换成图片进行Ai识别后合并成的并不存在音频，所以需要将原视频的音频提取（mp3只是音频格式，也可以使用其他的），在将获取到的视频和音频合并，就是我们想要的结果**

&nbsp; 
#### **问：python这么强大，不能把视频和音频合成吗？**
#### **答：之前试过sox用于合成视频，搞了一晚上也没弄好（菜是原罪），有兴趣的小伙伴可以自己尝试下（弄好的话记得给俺评论下，在线‘白嫖’，哈哈哈）**

&nbsp; 
#### **问：运行cv2ImgAddText报错OSError: cannot open resource**
#### **答：在Windows中，ImageFont.truetype会自动在系统路径中搜索字体名，而在Ubuntu中则不会。**
&nbsp; 
#### **问：生成的视频中识别中文是乱码**
#### **答：该脚本是在Windows10下调试的，后转到平台（平台内已调试，可以运行），字体不对或者修改其他字体，建议windows下运行或更改字体**

#### **注：替换字体，终端输入  `fc-list :lang=zh`**
#### **可以查看中文字体有什么，复制路径到脚本替换**

&nbsp; 
![](https://ai-studio-static-online.cdn.bcebos.com/2aeff9ffb9534c8e89f7f93874b8642a2251728209874659b470cbfede6b63b4)


### 个人主页：我在AI Studio上获得钻石等级，点亮9个徽章，来互关呀~[乌拉__----]( https://aistudio.baidu.com/aistudio/personalcenter/thirdview/311241)
### 简介：
#### **00年小菜鸟一枚**
#### **专业：嵌入式专业，专科在读**
#### **爱好：瞎搞，学习中...**
