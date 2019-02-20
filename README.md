# 音乐推荐系统
代码模块主要有三部分：

- 分析模块
    - `notebook`: 主要是原始数据进行分析，对数据进行处理
- 数据模块
    - `raw_data`: 音乐数据原始数据存放目录
        - `music_meta` 音乐数据759984条记录。字段说明：音乐id，简单描述信息desc，音乐时长total_timelen，location，音乐标签tags
        - `user_profile.data` 用户数据100000条记录。字段说明： 用户id,性别gender，年龄段age，salary薪资，地区省份
        - `user_watch_pref.xml` 用户行为数据321039条记录。字段说明： 用户id，音乐id，收听时长（单位秒），收听时间点（hour）

- **代码模块**
    - `recall`: 召回模块主要用的协同过滤做召回
    - `rank`: rank模块主要是recall传过来的数据做分析
    
## recall部分：召回/match
`item_base`和`user_base`是我们在协同过滤课程中已经实现了的，现在我们这里只是方法调用
- `item_base`: 基于物品的协同过滤
- `user_base`: 基于用户的协同过滤
- `config`: 所有数据存储输入输出的路径，以及原始数据
- `gen_cf_data`: 生成协同过滤需要用到的训练数据
- `cf_rec_list`: 离线的 实现利用`item_base`和`user_base`，线上一般存在redis中

## rank部分： LR模型训练，工程