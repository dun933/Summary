## MobileNet-SSD

<div align="center">
    <img  src="image/MobileNet_SSD.gif" />
</div>

&emsp;&emsp;实验在 KITTI 数据集上训练 MobileNet-SSD 模型，模型输入大小为 414*125，初始学习率为 0.0001，动量参数 0.9，权重衰减参数 0.0005，学习率在 60000，80000，90000次衰减 0.1，批处理量为 6，总迭代次数为 90000 次。训练结果如下表所示：

<table>
        <tr>
            <th>mAP</th>
            <th>Car/AP</th>
            <th>Pedestrian/AP</th>
            <th>Cyclist/AP</th>
            <th>Time</th>
        </tr>
        <tr>
            <th>67.3%</th>
            <th>82.2%</th>
            <th>57.7%</th>
            <th>61.7%</th>
            <th>165ms</th>
        </tr>
</table>

