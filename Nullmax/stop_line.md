## Stop-line detecting and tracking

<div align="center">
    <img  src="stop-line-dectect-and-tracking/demo/Stopline_tracking_demo.gif" />
</div>

### Object Detection

&emsp;&emsp;C + + 下 [g-darknet](https://github.com/generalized-iou/g-darknet) 框架的YOLOv3，硬件平台 GTX1080 8G 。

#### 配置参数

&emsp;&emsp;实验在 Stopline 数据集下，分别在 MSE loss 、GIoU loss 和 IoU loss 上训练770个epoch， Batch size = 128，Learning rate = 0.001，Normalizer (iou: 0.750000, cls: 1.000000)，使用 SGD 优化器。

#### 评估结果

&emsp;&emsp;测试数据为 Stopline 测试集 200 个样本，模型评估结果如下表所示：

<table>
        <tr>
            <th></th>
            <th>MSE Loss/mAP</th>
            <th>GIoU Loss/mAP</th>
            <th>IoU Loss/mAP</th>
        </tr>
        <tr>
            <th>IoU=0.50</th>
            <th>0.499</th>
            <th>0.668</th>
            <th>0.671</th>
        </tr>
    	<tr>
            <th>IoU=0.55</th>
            <th>0.438</th>
            <th>0.638</th>
            <th>0.592</th>
        </tr>
        <tr>
            <th>IoU=0.60</th>
            <th>0.335</th>
            <th>0.538</th>
            <th>0.483</th>
        </tr>
        <tr>
            <th>IoU=0.65</th>
            <th>0.200</th>
            <th>0.471</th>
            <th>0.375</th>
        </tr>
        <tr>
            <th>IoU=0.70</th>
            <th>0.124</th>
            <th>0.304</th>
            <th>0.254</th>
        </tr>
        <tr>
            <th>IoU=0.75</th>
            <th>0.055</th>
            <th>0.178</th>
            <th>0.117</th>
        </tr>
        <tr>
            <th>IoU=0.80</th>
            <th>0.029</th>
            <th>0.105</th>
            <th>0.062</th>
        </tr>
        <tr>
            <th>IoU=0.85</th>
            <th>0.017</th>
            <th>0.069</th>
            <th>0.030</th>
        </tr>
        <tr>
            <th>IoU=0.90</th>
            <th>0.011</th>
            <th>0.030</th>
            <th>0.005</th>
        </tr>
        <tr>
            <th>IoU=0.95</th>
            <th>0.000</th>
            <th>0.005</th>
            <th>0.000</th>
        </tr>
        <tr>
            <th>GIoU=0.50</th>
            <th>0.469</th>
            <th>0.664</th>
            <th>0.658</th>
        </tr>
        <tr>
            <th>GIoU=0.55</th>
            <th>0.358</th>
            <th>0.581</th>
            <th>0.575</th>
        </tr>
        <tr>
            <th>GIoU=0.60</th>
            <th>0.275</th>
            <th>0.512</th>
            <th>0.455</th>
        </tr>
        <tr>
            <th>GIoU=0.65</th>
            <th>0.162</th>
            <th>0.463</th>
            <th>0.343</th>
        </tr>
        <tr>
            <th>GIoU=0.70</th>
            <th>0.120</th>
            <th>0.291</th>
            <th>0.236</th>
        </tr>
        <tr>
            <th>GIoU=0.75</th>
            <th>0.048</th>
            <th>0.175</th>
            <th>0.108</th>
        </tr>
        <tr>
            <th>GIoU=0.80</th>
            <th>0.029</th>
            <th>0.105</th>
            <th>0.062</th>
        </tr>
        <tr>
            <th>GIoU=0.85</th>
            <th>0.017</th>
            <th>0.066</th>
            <th>0.030</th>
        </tr>
        <tr>
            <th>GIoU=0.90</th>
            <th>0.011</th>
            <th>0.030</th>
            <th>0.005</th>
        </tr>
   		<tr>
            <th>GIoU=0.95</th>
            <th>0.000</th>
            <th>0.005</th>
            <th>0.000</th>
        </tr>
</table>

### Object Tracking

&emsp;&emsp;实验采用 SORT 多目标跟踪算法对 Stop-line 进行 on-line tracking。

####  ***1. 简介***

&emsp;&emsp;SORT算法是以目标检测算法输出的 bbox 为基础，通过 Linear Kalman Filter 和 Hungary Algorithm 实现对目标的跟踪。其中 Kalman Filter 负责估计目标的运动信息。若没有检测框信息，即目标检测算法没有检测出目标，则使用线性模型进行位置预测；Hungary Algorithm 负责计算目标当前帧与上一帧的关联性，以此来管理追踪目标的生命周期。

#### ***2. 方法***

***检测阶段***

&emsp;&emsp;采用 ***YOLOv3*** 检测模型对 Stop-line 进行检测， 输出目标的 <img src="http://latex.codecogs.com/gif.latex?bbox(xmin,ymin,xmax,ymax)" />；

***跟踪阶段*** 

&emsp;&emsp;将 Stop-line 帧间运动近似为简单的线性运动通过 Kalman Filter 对下一帧的 bbox 位置进行预测，状态向量定义如下:

<div align="center">
    <img  src="http://latex.codecogs.com/gif.latex?{x}_{k}^-=[x_{min},v_{xmin},y_{min},v_{ymin},x_{max},v_{xmax},y_{max},v_{ymax}]^T" />
</div>
&emsp;&emsp;观测向量为:

<div align="center">
    <img src="http://latex.codecogs.com/gif.latex?z_k=[x_{min},y_{min},x_{max},y_{max}]^T" />
</div>
&emsp;&emsp;状态预测方程为：

<div align="center">
    <img src="http://latex.codecogs.com/gif.latex?\hat{x}_k^-=F\cdot\hat{x}_{k-1}^-+u" />
    <br/><br/>
    <img src="http://latex.codecogs.com/gif.latex?P_k^-=F\hat{P}_{k-1}F^T+Q" />
</div>

&emsp;&emsp;其中状态转移矩阵 F 为：
<div align="center">
    <img src="http://latex.codecogs.com/gif.latex?F=\left[\begin{matrix}1 & dt & 0 & 0 & 0 & 0 & 0 & 0\\0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 1 & dt & 0 & 0 & 0 & 0\\0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 1 & dt & 0 & 0\\0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 1 & dt\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\end{matrix}\right]" />
</div>

&emsp;&emsp;状态更新方程如下：

<div align="center">
    <img src="http://latex.codecogs.com/gif.latex?\hat{x}_k=\hat{x}_k^-+K_k(z_k-H\cdot\hat{x}_k^-)"/> <br/><br/>
    <img src="http://latex.codecogs.com/gif.latex?\hat{P}_k=P_k^--KHP_k^-"/> <br/><br/>
    <img src="http://latex.codecogs.com/gif.latex?K_k=\frac{\hat{P}_k^-H^T}{H\hat{P}_k^-H^T+R}"/>
</div>

&emsp;&emsp;噪声矩阵和 H 如下：

<div align="center">
	<img src="http://latex.codecogs.com/gif.latex?Q=\left[\begin{matrix}0.25 & 0.5 & 0 & 0 & 0 & 0 & 0 & 0\\0.5 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0.25 & 0.5 & 0 & 0 & 0 & 0\\0 & 0 & 0.5 & 1 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0.25 & 0.5 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0.25 & 0.5\\0 & 0 & 0 & 0 & 0 & 0 & 0.5 & 1\end{matrix}\right]"/><br/><br/>
	<img src="http://latex.codecogs.com/gif.latex?R=\left[\begin{matrix}10 & 0 & 0 & 0 \\0 & 10 & 0 & 0 \\0 & 0 & 10 & 0 \\0 & 0 & 0 & 10 \\\end{matrix}\right]"/><br/><br/>
    <img src="http://latex.codecogs.com/gif.latex?H=\left[\begin{matrix}1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\\end{matrix}\right]"/>
</div>

&emsp;&emsp;初始化状态：

<div align="center">
		<img src="http://latex.codecogs.com/gif.latex?P_0=\left[\begin{matrix}10 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 10 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 10 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 10 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 10 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 10 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 10 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 10\end{matrix}\right]"/>
</div>

&emsp;&emsp;使用 Hungary Algorithm 进行数据关联，使用的cost矩阵是原有目标在当前帧中的预测位置和当前帧目标检测框之间的 IoU，当小于指定 IoU 阈值结果是无效的。

***追踪目标的出现和消失***

&emsp;&emsp;检测到某个目标和所有已有目标预测结果的 bbox 的 IoU 都小于指定阈值时，则认为出现了新的待追踪目标，使用 bbox 初始化新目标的位置，速度设置为 0。新的追踪目标会经历一段待定时间去和检测结果进行关联以累计出现新目标的置信度，从而防止目标检测的虚警造成的新追踪目标误创建。 

&emsp;&emsp;在实验中，由于远处的 Stop-line 的 bbox 非常小，会使下一帧中同一目标的 IoU 值为 0 从而似的算法判定为是新的目标而失去跟踪，因此采用了可以更好的衡量距离的 GIoU 为判定数据关联的标准。

&emsp;&emsp;如果连续 <img src="http://latex.codecogs.com/gif.latex?T_{lost}"/>  帧没有实现已追踪目标预测位置和检测框的 GIoU 匹配，则认为目标消失。实验中设置<img src="http://latex.codecogs.com/gif.latex?T_{lost}=1"/>，原因有二，一是匀速运动假设不合理，二是作者主要关注短时目标追踪。另外，尽早删除已丢失的目标有助于提升追踪效率。
