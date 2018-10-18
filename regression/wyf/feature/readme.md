## 特征处理说明

这一部分说明“Tags”列的处理方法。下图展示了原始数据的样子。

![](http://bit.ly/2yMcfrj)

### TripType(旅游类型)

|类型|值|
|-|-|
|Leisure trip| 0|
|Business trip|1|
|无此类标签|-1|

### traveler_type（旅游者类型）

|类型|值|
|-|-|
|Solo traveler| 0|
|Couple|1|
|Family with young children|2|
|Family with older children|3|
|Travelers with friends|4|
|Group|5|
|无此类标签|-1|

### order_type(订单类型)

这一个订单类型说明的是，该订单是否使用移动设备发起提交。

|类型|值|
|-|-|
|Submitted from a mobile device| 1|
|无此类标签|0|

### nights_num(居住天数)

就，对于下面的这类信息进行处理，提取了订酒店的顾客在酒店居住的天数。

![](http://bit.ly/2ynoLyh)

### with_pet(携带宠物)

一些顾客携带了宠物，因此会有一个“with a pet"的标签。

|类型|值|
|-|-|
|With a pet| 1|
|无此类标签|0|

## 暂未处理特征

房间类型

![](http://bit.ly/2yLJXNM)

