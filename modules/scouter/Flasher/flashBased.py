import time
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class FlashBased():
    '''
        extContent기반으로 kmeans clustering을 통하여 기사를 분류한 후
        중복성을 검출하는 프로그램입니다.
    '''

    def __init__(self, num_cluster, sim_rate, docs):
        self.num_cluster = num_cluster
        self.sim_rate = sim_rate
        self.docs = docs
        self.fb_result = []

    def runFlash(self):
        start = time.time()  # whole, entire

        # preprocessing
        df_data = pd.DataFrame([self.docs[x] for x in range(0, len(self.docs))])
        #이주영 추가
        #df_data.SentimentText=df_data.SentimentText.astype(str)

        sentence = ""
        exceptSymbol = (
        'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'NF', 'NV', 'NA', 'ETM', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
        'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM')
        for data, nid in zip(df_data['analyzed_text'], df_data['news_id']):
            for ssize in range(0, len(data['sentence'])):
                for data2 in data['sentence'][ssize]['WSD']:
                    if data2['type'] not in exceptSymbol:
                        sentence = sentence + data2['text'] + " "

            index = df_data.loc[df_data['news_id'] == nid].index.values[0]
            df_data.at[index, 'sentence_text'] = sentence
            sentence = ""
        df_data = df_data.drop("analyzed_text", 1)

        # K-Means
        tvec = TfidfVectorizer()
        #pandas float error
        df_data['sentence_text']=df_data['sentence_text'].astype(str)
        X = tvec.fit_transform(df_data['sentence_text'])
        model = KMeans(n_clusters=self.num_cluster, init='k-means++', max_iter=7, n_init=5)
        tk_label = model.fit_predict(X)
        df_data["cluster_num"] = tk_label

        # 각 클러스터 별로 배열 만들기
        td = []
        each_cluster = []
        # initial
        for i in range(0, self.num_cluster):
            each_cluster.append([])
            td.append(None)

        for cnum, nid, content in zip(df_data['cluster_num'], df_data['news_id'], df_data['extContent']):
            each_cluster[cnum].append((nid, content))

        # 각 클러스터 별로 pandas만들기
        pd_cluster = []
        for row in each_cluster:
            pd_cluster.append(pd.DataFrame(row, columns=['news_id', 'Content']))

        # countVectorizer
        count_matrix = []
        cvec = CountVectorizer()
        for i in range(0, self.num_cluster):
            count_matrix.append(cvec.fit_transform(pd_cluster[i]['Content']))

        start_cos = time.time()
        pd_result = pd.DataFrame(columns=["A", "B"])
        resultindex = 0
        # cosine similarity
        for index in range(0, self.num_cluster):
            cosine_sim = cosine_similarity(count_matrix[index], count_matrix[index])
            # print(cosine_sim)
            # print(index,": ",cosine_sim.shape)
            num_doc = count_matrix[index].shape[0]
            for i in range(0, num_doc):
                for x in range(i + 1, num_doc):
                    if (cosine_sim[i][x] > self.sim_rate):
                        if (i != x):
                            pd_result.loc[resultindex] = (
                            pd_cluster[index]['news_id'][i], pd_cluster[index]['news_id'][x])
                            resultindex += 1

        print("elapsed time taken for flash: ", time.time() - start, "seconds.")
        elspedTime = time.time() - start

        self.fb_result = pd_result

        return elspedTime, pd_result

    def eliminateDup(self):
        def isExistin(list2D, value):
            index = 0
            for list1D in list2D:
                if value in list1D:
                    return index
                index += 1
            return -1

        dlist = []
        dlist.append([])
        for (first, second) in zip(self.fb_result['A'], self.fb_result['B']):
            if (isExistin(dlist, first) != -1):
                index = isExistin(dlist, second)
                if (index == -1):
                    dlist[index].append(second)
            else:
                index2 = isExistin(dlist, second)
                if (index2 != -1):
                    dlist[index2].append(first)
                else:
                    dlist.append([first, second])

        deleteIt = []
        for index1 in range(1, len(dlist)):
            for index2 in range(1, len(dlist[index1])):
                deleteIt.append(dlist[index1][index2])

        return deleteIt
