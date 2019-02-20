import recall.config as conf
import math

a = conf.a
# user_id = '014c8e555aa35acfb6b7008a01e085f2'
user_id = '018de1e53b639c0b45fcb7362b01c027'

# 以前听什么歌
user_watch = conf.gen_user_watch()
music_df = conf.gen_music_meta()
df = user_watch.merge(music_df, how='inner', on='music_id')
del user_watch
df = df.loc[df['user_id'] == user_id, ['music_name']]
print(df.head(100))
# step1: 载入特征处理
# load user and item category feature
with open(conf.user_feat_map_file, 'r', encoding='utf-8') as f:
    category_feat_dict = eval(f.read())

# load cross feature
with open(conf.cross_file, 'r', encoding='utf-8') as cross_file:
    cross_feat_dict = eval(cross_file.read())

# step 2: 载入model
# load LR model
with open(conf.model_file, 'r', encoding='utf-8') as model_file:
    model_dict = eval(model_file.read())
w = model_dict['w']
b = model_dict['b']

# step 3: match/recall(协同过滤，召回候选集)
rec_item_all = dict()
# 3.1 CF
# 3.1.1 user base recall
with open(conf.cf_rec_lst_outfile, 'r', encoding='utf-8') as rec_lst_file:
    cf_rec_lst = eval(rec_lst_file.read())
key = conf.UCF_PREFIX + user_id
# 用户协同召回的物品集合
ucf_rec_lst = cf_rec_lst[key]

for item, score in ucf_rec_lst:
    rec_item_all[item] = float(score) * a

# 3.1.2 item base recall
key = conf.ICF_PREFIX + user_id
# 物品协同召回的物品集合
icf_rec_lst = cf_rec_lst[key]

for item, score in icf_rec_lst:
    if rec_item_all.get(item, -1) == -1:
        rec_item_all[item] = float(score) * (1 - a)
    else:
        # 当两种推荐中物品相同时，求和
        rec_item_all[item] += float(score) * (1 - a)


# 3.2 CB

# step 4: 调取用户的物品的服务
# 4.1 用户属性
user_df = conf.gen_user_profile()
age, gender, salary, province = '', '', '', ''
for _, row in user_df.loc[user_df['user_id'] == user_id, :].iterrows():
    age, gender, salary, province = row['age'], row['gender'], row['salary'], row['province']
    (age_idx, gender_idx, salary_idx, province_idx) = (category_feat_dict['age_' + age],
                                                      category_feat_dict['gender_' + gender],
                                                      category_feat_dict['salary_' + salary],
                                                      category_feat_dict['province_' + province])
    print('age : ' + age, 'gender : ' + gender, 'salary : ' + salary, 'province : ' + province)

del user_df

rec_lst = []
for music_id in rec_item_all.keys():
    # item_df = conf.gen_music_meta()
    music_loc, item_name = '', ''
    for _, row in music_df.loc[music_df['music_id'] == int(music_id), :].iterrows():
        # print(row)
        music_loc, item_name = row['music_loc'], row['music_name']
        # print('music_loc: %s,music_name: %s' % (music_loc, item_name))
    location_idx = category_feat_dict['music_loc_' + music_loc]

    ui_key = user_id + '_' + music_id
    cross_value = float(cross_feat_dict.get(ui_key, 0))

    # predict
    # y = sigmoid(-wx-b)
    wx_score = float(b)
    wx_score += w[age_idx] + w[gender_idx] + w[salary_idx] + w[province_idx] + w[location_idx]
    wx_score += w[-1] * cross_value
    # sigmoid:p(y=1|x)=(1+exp(-wx)
    final_rec_score = 1/(1 + math.exp(-wx_score))
    score = rec_item_all[music_id]
    final_rec_score = 0.3 * final_rec_score + 0.7 * score

    rec_lst.append((music_id, item_name, final_rec_score))

# setp 5: 排序
rec_sort_list = sorted(rec_lst, key=lambda x: x[2], reverse=True)

# step 6: topN 取5个
rec_filter_list = rec_sort_list[:5]

# step 7: 返回数据+包装
return_list = [' => '.join([i_id, name, str(score)]) for i_id, name, score in rec_filter_list]
print('\n'.join(return_list))
