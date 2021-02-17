import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def exportcsv(pred):
    res = pd.DataFrame(columns=['PassengerId','Survived'])
    res['PassengerId'] = test_data['PassengerId']
    res['Survived'] = pd.Series(pred)
    res.to_csv('~/Desktop/res.csv', index=False)

# action 1
# 加载
train_data = pd.read_csv('./Titanic_Data/train.csv')
test_data = pd.read_csv('./Titanic_Data/test.csv')

# # 数据探索
# print(train_data.info())
# print('-'*30)
# print(train_data.describe())
# print('-'*30)
# print(train_data.describe(include=['O']))
# print('-'*30)

# 预处理
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

# 把字典中一些类别数据 Pclass Sex Embarked，分别进行转化成特征
dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))
print(dvec.feature_names_)
print(train_features.shape)

# cross validation - split the training set to train and test
train_features_X_train, train_features_X_test, train_labels_Y_train, train_labels_Y_test = train_test_split(train_features, train_labels, test_size=0.33, random_state=42)

# -------------------------------------
# 方法一 LOGISTIC REGRESSION
# -------------------------------------
# cross validation
clf1_cv = LogisticRegression(max_iter=100, verbose=True, random_state=33, tol=1e-4)
clf1_cv.fit(train_features_X_train, train_labels_Y_train)
print(u'Cross Validation LOGISTIC REGRESSION score 准确率为 %.4lf' \
      % round(clf1_cv.score(train_features_X_test, train_labels_Y_test), 6))
pred1_cv = clf1_cv.predict(test_features)
exportcsv(pred1_cv)

# non-cross validation
clf1 = LogisticRegression(max_iter=100, verbose=True, random_state=33, tol=1e-4)
clf1.fit(train_features, train_labels)
print(u'Non-Cross Validation LOGISTIC REGRESSION score 准确率为 %.4lf' \
      % clf1.score(train_features, train_labels))
pred1 = clf1.predict(test_features)
# -------------------------------------
# 方法二 SVM
# -------------------------------------
# cross validation
clf2_cv = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=1e-4, verbose=False)
clf2_cv.fit(train_features_X_train, train_labels_Y_train)
print(u'Cross Validation SVM score 准确率为 %.4lf' \
      % round(clf2_cv.score(train_features_X_test, train_labels_Y_test), 6))
pred2_cv = clf2_cv.predict(test_features)
exportcsv(pred2_cv)

# non-cross validation
clf2 = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=1e-4, verbose=False)
clf2.fit(train_features, train_labels)
print(u'Non-Cross Validation SVM score 准确率为 %.4lf' \
      % clf2.score(train_features, train_labels))
pred2 = clf2.predict(test_features)
# -------------------------------------
# 方法三 决策树
# -------------------------------------
# # cross validation
# clf3_cv = DecisionTreeClassifier(criterion='entropy')
# clf3_cv.fit(train_features_X_train, train_labels_Y_train)
# print(u'Cross Validation 决策树 score 准确率为 %.4lf' \
#       % round(clf3_cv.score(train_features_X_test, train_labels_Y_test), 6))
# pred3_cv = clf3_cv.predict(test_features)
# exportcsv(pred3_cv)
#
# # non-cross validation
# clf3 = DecisionTreeClassifier(criterion='entropy')
# clf3.fit(train_features, train_labels)
# print(u'Non-Cross Validation 决策树 score 准确率为 %.4lf' \
#       % clf3.score(train_features, train_labels))
# pred3 = clf3.predict(test_features)
# -------------------------------------
# 方法四 随机森林
# -------------------------------------
# # cross validation
# clf4_cv = RandomForestClassifier(n_estimators=200, max_depth=5)
# clf4_cv.fit(train_features_X_train, train_labels_Y_train)
# print(u'Cross Validation 随机森林 score 准确率为 %.4lf' \
#       % round(clf4_cv.score(train_features_X_test, train_labels_Y_test), 6))
# pred4_cv = clf4_cv.predict(test_features)
# exportcsv(pred4_cv)
#
# # non-cross validation
# clf4 = RandomForestClassifier(n_estimators=200, max_depth=5)
# clf4.fit(train_features, train_labels)
# print(u'Non-Cross Validation 随机森林 score 准确率为 %.4lf' \
#       % clf4.score(train_features, train_labels))
# pred4 = clf4.predict(test_features)
# -------------------------------------

#
# # action 2
# def get_page_content(request_url):
#     headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
#     html=requests.get(request_url,headers=headers,timeout=10)
#     content = html.text
#     soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')
#     return soup
#
# def analysis(soup):
#     temp = soup.find('div',class_="tslb_b")
#     df = pd.DataFrame(columns = ['id', 'brand', 'car_model', 'type', 'desc', 'problem', 'datetime', 'status'])
#     tr_list = temp.find_all('tr')
#     for tr in tr_list:
#         td_list = tr.find_all('td')
#         if len(td_list) > 0:
#             id, brand, car_model, type, desc, problem, datetime, status = td_list[0].text, td_list[1].text, td_list[2].text, td_list[3].text, td_list[4].text, td_list[5].text, td_list[6].text,td_list[7].text
#             temp = {}
#             temp['id'] = id
#             temp['brand'] = brand
#             temp['car_model'] = car_model
#             temp['type'] = type
#             temp['desc'] = desc
#             temp['problem'] = problem
#             temp['datetime'] = datetime
#             temp['status'] = status
#             df = df.append(temp, ignore_index=True)
#     return df
#
# result = pd.DataFrame(columns=['id', 'brand', 'car_model', 'type', 'desc', 'problem', 'datetime', 'status'])
# base_url = 'http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-0-0-0-0-0-1.shtml'
# request_url = base_url
# soup = get_page_content(request_url)
# df=analysis(soup)
# result = result.append(df)
#
# print(result)
# result.to_excel('./car_complaint.xlsx', index=False)