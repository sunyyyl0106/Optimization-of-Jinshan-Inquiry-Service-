import jieba
import pandas as pd
import os
import shutil
import pickle  # 持久化
from numpy import *
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets._base import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法
import warnings

# 目录位置
Directory = 'C:/Users/Sun/Downloads/Train_Data'
# 数据集csv名称
csv_name = '中天社区去紧急通知csv.csv'
# 测试集csv名称
test_csv_name = '中天社区去紧急通知csv.csv'
# csv列名
col_name = ['desc', 'case_name', 'grid_name', 'FROM_UNIXTIME(a.create_time)', 'reporter_name', 'reporter_mobile']
# 分类列名
cat_name = 'case_name'


def readFile(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content


def saveFile(path, result):
    with open(path, 'w', errors='ignore') as file:
        file.write(result)


def segText(inputPath, resultPath, sentence_in):
    if sentence_in == 'nan':
        fatherLists = os.listdir(inputPath)  # 主目录
        for eachDir in fatherLists:  # 遍历主目录中各个文件夹
            eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
            each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
            if not os.path.exists(each_resultPath):
                os.makedirs(each_resultPath)
            childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件

            for eachFile in childLists:  # 遍历每个文件夹中的子文件
                eachPathFile = eachPath + eachFile  # 获得每个文件路径
                # print(eachFile)
                # print(each_resultPath)
                content = readFile(eachPathFile)  # 调用上面函数读取内容
                # content = str(content)
                result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
                # result = content.replace("\r\n","").strip()

                cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
                saveFile(each_resultPath + eachFile, " ".join(cutResult))  # 调用上面函数保存文件
    else:
        pre_path = resultPath + '街面秩序/'
        if not os.path.exists(pre_path):
            os.makedirs(pre_path)
        result = (str(sentence_in)).strip('[\'').rstrip('\']')
        saveFile(pre_path + 'predict.txt', result)

def bunchSave(inputFile, outputFile):
    catelist = os.listdir(inputFile)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
    for eachDir in catelist:
        eachPath = inputFile + eachDir + "/"
        fileList = os.listdir(eachPath)
        for eachFile in fileList:  # 二级目录中的每个子文件
            fullName = eachPath + eachFile  # 二级目录子文件全路径
            filename = open(fullName)
            line = filename.readline()
            while line:
                bunch.label.append(eachDir)  # 当前分类标签
                bunch.filenames.append(fullName)  # 保存当前文件的路径
                bunch.contents.append(line.strip())  # 保存文件词向量
                line = filename.readline()

    with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)


def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
    return bunch


def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)


def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    # print(stopWordList)
    return stopWordList


def getTFIDFMat(inputPath, stopWordList, outputPath):  # 求得TF-IDF向量
    bunch = readBunch(inputPath)
    #    print(bunch)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.9)
    transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    # print(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_
    # print(tfidfspace)
    writeBunch(outputPath, tfidfspace)


def getTestSpace(inputPath, testSetPath, trainSpacePath, stopWordList, testSpacePath):
    bunch1 = readBunch(inputPath)
    bunch2 = readBunch(testSetPath)
    # print(bunch2)
    tfidfspace = Bunch(target_name=bunch1.target_name, label=bunch1.label, filenames=bunch1.filenames, tdm=[],
                       vocabulary={})
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch2.target_name, label=bunch2.label, filenames=bunch2.filenames, tdm=[],
                      vocabulary={})
    # 导入训练集的词袋
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.9)
    # trainbunch = readBunch(trainSpacePath)
    # # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    # vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.9,
    #                              vocabulary=trainbunch.vocabulary)

    tfidfspace.tdm = vectorizer.fit_transform(bunch1.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_

    testSpace.tdm = vectorizer.transform(bunch2.contents)
    testSpace.vocabulary = vectorizer.vocabulary_
    # print(testSpace.tdm.todense())
    df = pd.DataFrame(testSpace.tdm.todense())
    df.to_csv(Directory + '/tdm transform.csv', index=False, header=False)
    # testSpace.vocabulary = trainbunch.vocabulary
    # 持久化
    writeBunch(testSpacePath, testSpace)


def bayesAlgorithm(trainPath, testPath):
    trainSet = readBunch(trainPath)
    testSet = readBunch(testPath)
    # print(testSet)
    clf = MultinomialNB(alpha=0).fit(trainSet.tdm, trainSet.label)
    predicted = clf.predict(testSet.tdm)
    if mode == '1':
        print(shape(trainSet.tdm))
        print(shape(testSet.tdm))
        for targetName in testSet.target_name:
            rate = 0
            total = 0
            for flabel ,filename, expct_cate in zip(testSet.label, testSet.filenames, predicted):
                if filename.split('/')[-1] == targetName + '.txt':
                    total += 1
                    # print("f:" + str(flabel))
                    # print("e:" + str(expct_cate))
                    if flabel != expct_cate:
                        rate += 1
                # print(filename, ":实际类别：", flabel, "-->预测类别：", expct_cate)
            print(targetName, "error rate:", float(rate) * 100 / float(total), "%")
            print('error数' + str(rate))
            print('本类别总数' + str(total))
        total = len(predicted)
        rate = 0
        for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
            if flabel != expct_cate:
                rate += 1
                # print(fileName, ":实际类别：", flabel, "-->预测类别：", expct_cate)
        print("总 error rate:", float(rate) * 100 / float(total), "%")
    if mode == '2':
        print('预测结果：')
        print(predicted)


def readCsv(name, Path):
    # 读取csv
    data = pd.read_csv(Directory + '/' + name, encoding='GB18030')
    data.columns = col_name
    j = 0
    grouped = data.groupby(cat_name)
    ind_frame = data.drop_duplicates(subset=[cat_name])

    # 按case_name输出txt
    str_name=str(cat_name)
    for str_name in ind_frame[cat_name]:
        j += 1
        print("第" + '{}'.format(j)+'个类别done')
        dir_path = Directory + Path + u"%s" % str_name
        os.mkdir(dir_path)
        txt_output_path = Directory + Path + u"%s" % str_name + '/' + u"%s" % str_name + ".txt"
        # print(txt_output_path)
        with open(txt_output_path, 'a+', encoding='ansi', errors='ignore') as f:
             for line in data[data[cat_name] == str_name].values:
                f.write(str(line[0]) + ' ' + str(line[2]).strip().replace('\n', '').replace('\r', '') + '\n')


warnings.filterwarnings("ignore")
jieba.setLogLevel(jieba.logging.INFO)
# 选择模式
print('请输入测试模式(1:测试集 2：实时预测):')
mode = input()
# 更新目录
if os.path.exists(Directory + "/data"):
    shutil.rmtree(Directory + "/data")
os.mkdir(Directory + "/data")
if os.path.exists(Directory + "/test_data"):
    shutil.rmtree(Directory + "/test_data")
os.mkdir(Directory + "/test_data")
if os.path.exists(Directory + "/segResult"):
    shutil.rmtree(Directory + "/segResult")
if os.path.exists(Directory + "/test_segResult"):
    shutil.rmtree(Directory + "/test_segResult")
print('更新目录done')
# 读取csv
readCsv(csv_name, '/data/')
print('读取csv数据集done')

# 数据集分词
segText(Directory + "/data/", Directory + "/segResult/", 'nan')
bunchSave(Directory + "/segResult/", Directory + "/train_set.dat")  # 输入分词，输出分词向量
print('数据集分词done')
# TF-IDF
stopWordList = getStopWord(Directory + "/stopwords/hit_stopwords.txt")  # 获取停用词
getTFIDFMat(Directory + "/train_set.dat", stopWordList, Directory + "/tfidfspace.dat")  # 输入词向量，输出特征空间
print('TF-IDFdone')
if mode == '1':
    # 读取测试集csv
    readCsv(test_csv_name, '/test_data/')
    print('读取csv测试集done')
    # 测试集分词
    segText(Directory + "/test_data/", Directory + "/test_segResult/", 'nan')  # 分词
    bunchSave(Directory + "/test_segResult/", Directory + "/test_set.dat")
    print('测试集分词done')
    # 测试
    print('测试结果:')
    getTestSpace(Directory + "/train_set.dat", Directory + "/test_set.dat", Directory + "/tfidfspace.dat", stopWordList, Directory + "/testspace.dat")
    bayesAlgorithm(Directory + "/tfidfspace.dat", Directory + "/testspace.dat")
if mode == '2':
    while 1:
        print('请输入需要预测的文本:')
        a = input()
        if a == "结束":
            break
        else:
            print(str(' '.join(jieba.cut(a))))
            sentence_in = str([' '.join(jieba.cut(a))])
            segText(Directory + "/test_data/", Directory + "/test_segResult/", sentence_in)  # 分词
            bunchSave(Directory + "/test_segResult/", Directory + "/test_set.dat")
            print('测试集分词done')
            # 测试
            print('测试结果:')
            getTestSpace(Directory + "/train_set.dat", Directory + "/test_set.dat", Directory + "/tfidfspace.dat", stopWordList, Directory + "/testspace.dat")
            bayesAlgorithm(Directory + "/tfidfspace.dat", Directory + "/testspace.dat")