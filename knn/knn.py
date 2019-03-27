import random
import csv

#读取csv  第二、第三列为描述癌细胞的两个特征，意为肿块密度、细胞大小，第四列为细胞类别（0为恶性，1为良性）
with open('breast-cancer-train.csv','r') as file:
    reader = csv.DictReader(file)
    # for row in reader:
    #     print(row)   
    datas = [row for row in reader]
#print(datas)

random.shuffle(datas)
n = len(datas)//3
#分组 数据一切为二 一组做训练 一组做测试

test_set = datas[0:n]
train_set = datas[n:]

#knn
#距离
def distance(d1,d2):
    res = 0
    for key in ('ClumpThickness','CellSize'):
        res+=(float(d1[key])-float(d2[key]))**2
    return res**0.5   

K = 50        
def knn(data):
    #1.求所有数据的距离
    res = [
        {"result":train['Type'],"distance":distance(data,train)}
        for train in train_set
    ]
    #2排序
    res = sorted(res,key=lambda item: item['distance'])
    #3取前k个数据
    res2 = res[0:K]
    #加权平均
    result = {'0':0,'1':0}
    #总距离 sum要做除数为了不报错初始值设置为1
    sum = 1
    for r in res2:
        sum += r['distance']

    for r in res2:
        result[r['result']]+=1-r['distance']/sum
    #print(result)
    #print(data['Type'])
    
    if result['1']>result['0']:
        return '1'
    else:
        return '0' #恶性   

#测试
#knn(test_set[0])
correct = 0
for test in test_set:
    result = test['Type']
    result2 = knn(test)

    if result == result2:
        correct+=1
#print(correct)
#print(len(test_set))
print("{:.2f}%".format(100*correct/len(test_set)))
