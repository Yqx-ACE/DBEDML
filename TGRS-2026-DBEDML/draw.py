import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#FFEBCD', '#FFF8DC', '#000000','#DC143C', '#B8860B', '#8B0000', '#00CED1', '#DCDCDC']  # 7个类，准备7种颜色



def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    x: dict = {}
    y: dict = {}
    
    for i in range(np.max(label)+1):
        x[f"{i + 1}"] = []
        y[f"{i + 1}"] = []
    for i in range(data.shape[0]):
        cc = label[i] + 1
        x[f"{cc}"].append(data[i, 0])
        y[f"{cc}"].append(data[i, 1])
    if np.max(label) == 16:
        A = plt.scatter(x[f"{1}"], y[f"{1}"], color=color_map[1], marker='o', s=0.5)
        B = plt.scatter(x[f"{2}"], y[f"{2}"], color=color_map[2], marker='o', s=0.5)
        C = plt.scatter(x[f"{3}"], y[f"{3}"], color=color_map[3], marker='o', s=0.5)
        D = plt.scatter(x[f"{4}"], y[f"{4}"], color=color_map[4], marker='o', s=0.5)
        E = plt.scatter(x[f"{5}"], y[f"{5}"], color=color_map[5], marker='o', s=0.5)
        F = plt.scatter(x[f"{6}"], y[f"{6}"], color=color_map[6], marker='o', s=0.5)
        G = plt.scatter(x[f"{7}"], y[f"{7}"], color=color_map[7], marker='o', s=0.5)
        H = plt.scatter(x[f"{8}"], y[f"{8}"], color=color_map[8], marker='o', s=0.5)
        I = plt.scatter(x[f"{9}"], y[f"{9}"], color=color_map[9], marker='o', s=0.5)
        J = plt.scatter(x[f"{10}"], y[f"{10}"], color=color_map[10], marker='o', s=0.5)
        K = plt.scatter(x[f"{11}"], y[f"{11}"], color=color_map[11], marker='o', s=0.5)
        L = plt.scatter(x[f"{12}"], y[f"{12}"], color=color_map[12], marker='o', s=0.5)
        M = plt.scatter(x[f"{13}"], y[f"{13}"], color=color_map[13], marker='o', s=0.5)
        N = plt.scatter(x[f"{14}"], y[f"{14}"], color=color_map[14], marker='o', s=0.5)
        O = plt.scatter(x[f"{15}"], y[f"{15}"], color=color_map[15], marker='o', s=0.5)
        P = plt.scatter(x[f"{16}"], y[f"{16}"], color=color_map[16], marker='o', s=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.legend((A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P), (
        'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16'),
                   loc="upper right")
        plt.savefig('speformertsne_sa.png')
    elif np.max(label) + 1 == 11:
        A = plt.scatter(x[f"{1}"], y[f"{1}"], color=color_map[1], marker='o', s=0.5)
        B = plt.scatter(x[f"{2}"], y[f"{2}"], color=color_map[2], marker='o', s=0.5)
        C = plt.scatter(x[f"{3}"], y[f"{3}"], color=color_map[3], marker='o', s=0.5)
        D = plt.scatter(x[f"{4}"], y[f"{4}"], color=color_map[4], marker='o', s=0.5)
        E = plt.scatter(x[f"{5}"], y[f"{5}"], color=color_map[5], marker='o', s=0.5)
        F = plt.scatter(x[f"{6}"], y[f"{6}"], color=color_map[6], marker='o', s=0.5)
        G = plt.scatter(x[f"{7}"], y[f"{7}"], color=color_map[7], marker='o', s=0.5)
        H = plt.scatter(x[f"{8}"], y[f"{8}"], color=color_map[8], marker='o', s=0.5)
        I = plt.scatter(x[f"{9}"], y[f"{9}"], color=color_map[9], marker='o', s=0.5)
        J = plt.scatter(x[f"{10}"], y[f"{10}"], color=color_map[10], marker='o', s=0.5)
        K = plt.scatter(x[f"{11}"], y[f"{11}"], color=color_map[11], marker='o', s=0.5)
        plt.xticks([])
        plt.yticks([])
       # plt.legend((A, B, C, D, E, F, G, H, I, J, K),('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11'),loc="upper right")
        plt.savefig(title+'.png')
    elif np.max(label) + 1 == 9:
        plt.figure(figsize=(14, 14))
        A = plt.scatter(x[f"{1}"], y[f"{1}"], color=color_map[1], marker='o', s=2.5)
        B = plt.scatter(x[f"{2}"], y[f"{2}"], color=color_map[2], marker='o', s=2.5)
        C = plt.scatter(x[f"{3}"], y[f"{3}"], color=color_map[3], marker='o', s=2.5)
        D = plt.scatter(x[f"{4}"], y[f"{4}"], color=color_map[4], marker='o', s=2.5)
        E = plt.scatter(x[f"{5}"], y[f"{5}"], color=color_map[5], marker='o', s=2.5)
        F = plt.scatter(x[f"{6}"], y[f"{6}"], color=color_map[6], marker='o', s=2.5)
        G = plt.scatter(x[f"{7}"], y[f"{7}"], color=color_map[7], marker='o', s=2.5)
        H = plt.scatter(x[f"{8}"], y[f"{8}"], color=color_map[8], marker='o', s=2.5)
        I = plt.scatter(x[f"{9}"], y[f"{9}"], color=color_map[9], marker='o', s=2.5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(title+'.png')

def main(data, label, name):
    n_samples, n_features = data.shape  # 根据自己的路径合理更改

    print('Begining......')

    # 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    result_2D = tsne_2D.fit_transform(data)

    print('Finished......')
    plot_embedding_2D(result_2D, label, name)  # 将二维数据用plt绘制出来
    
import numpy as np
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D

def main2(data, label, name):
    data_sne_src = np.copy(data)
    label_data_Src = np.copy(label)
    X_tsne_src = umap.UMAP(n_components=2, random_state=100).fit_transform(data_sne_src)
    
    # 定义颜色列表
    colors = [
        '#FF0000', '#FF4500', '#D2691E', '#DEB887', '#FFA500', '#FFD700',
        '#808080', '#9656DB', '#7CFC00', '#008000', '#00FF7F', '#00FFFF',
        '#1E90FF', '#0000FF', '#7B68EE', '#FF1493'
    ]
    
    # 创建绘图
    plt.figure(figsize=(14, 14))
    
    # 为每个类别添加散点图
    for cla_label in range(0, np.max(label_data_Src) + 1):
        plt.scatter(X_tsne_src[label_data_Src == cla_label][:, 0], 
                    X_tsne_src[label_data_Src == cla_label][:, 1],
                    c=colors[cla_label - 1], marker='o', s=1.5)
        print('class', cla_label, np.count_nonzero(X_tsne_src[label_data_Src == cla_label][:, 0]), '\n')

    # 自定义图例
    legend_elements = []
    for i, cla_label in enumerate(range(0, np.max(label_data_Src) + 1)):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[cla_label - 1], markersize=10, label=f'Class {cla_label+1}'))
        
    
    # 添加图例
    plt.legend(handles=legend_elements,loc = "upper right")
    
    # 保存图像
    plt.savefig(name + str(label.shape[0]) + '.png', dpi=300)

# 假设数据已经准备好并调用此函数


if __name__ == "__main__":
    output = np.load('output.npy')
    output = output.reshape(-1, output.shape[2])
    data_gt = data_reader.Houston().truth
    data_gt = data_gt.astype('int').flatten()
    x, y = [], []
    for i in range(data_gt.shape[0]):
        if data_gt[i] != 0:
            x.append(output[i, :])
            y.append(data_gt[i])
    x = np.array(x)
    y = np.array(y)
    main(x, y)