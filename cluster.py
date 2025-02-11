import sys
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def get_paths(file) -> np.ndarray:
    with open(file) as f:
        paths = f.readlines()
        # return np.array(["./training_samples/" + path.rstrip() for path in paths])
        return np.array([path.rstrip() for path in paths])


def get_images(paths: np.ndarray):
    ret = []
    for path in paths:
        with Image.open(path) as img:
            greyscale = ImageOps.grayscale(img)
            ret.append(np.asarray(greyscale))
    return ret


'''
    Crop and scale images:
    input: images - array containing numpy 2D array of 
                    greyscale pixel values 0-225
    output: array of normalized 2D arrays
'''
def normalize_images(images: np.ndarray, eps=10, resize_to=60):
    WHITE = 255
    normalized_images = []

    for i in range(len(images)):
        img = images[i]

        n_rows = img.shape[0]
        n_cols = img.shape[1]
    
        # columns left to right:
        for col in range (n_cols):
            if WHITE - np.min(img[:,col]) < eps:
                img = np.delete(img, col, axis=1)
            else:
                break

        # columns right to left:
        for col in range (n_cols - 1, -1, -1):
            if WHITE - np.min(img[:,col]) < eps:      # if entire column is white (- epsilon),
                img = np.delete(img, col, axis=1)     # then remove it
            else:
                break

        # rows top to bottom
        for row in range(n_rows):
            if WHITE - np.min(img[row]) < eps:
                img = np.delete(img, row, axis=0)
            else:
                break
            
        # rows bottom to top
        for row in range(n_rows -1, -1, -1):
            if WHITE - np.min(img[row]) < eps:
                img = np.delete(img, row, axis=0)
            else:
                break

        pillow_img = Image.fromarray(img, 'L')
        resized_pillow_img = pillow_img.resize((resize_to, resize_to), Image.LANCZOS)
        resized_img = np.asarray(resized_pillow_img)
        flat_img = resized_img.flatten()
        flat_img = np.concatenate((flat_img, [img.shape[0], img.shape[1], img.shape[0]/img.shape[1]]))
        normalized_images.append(flat_img)

    return StandardScaler().fit_transform(np.array(normalized_images))


def create_output_list(labels, paths):
    lists = [[] for _ in range(len(set(labels)))]
    for i in range(labels.shape[0]):
        lists[labels[i]].append(paths[i])
    return lists


'''
    Create an HTML file with grouped images
'''
def create_output_files(lists):
    with open("output.txt", "w") as out:
        for j in range(len(lists)):
            line = lists[j]
            for i in range(len(line)):
                name = line[i].split('/')[-1]
                out.write(name)
                if i is len(line) - 1:
                    out.write("\n")
                else:
                    out.write(" ")

    with open("index.html", "w") as out:
        out.write("<!DOCTYPE html>\n")
        out.write("<html>\n")
        out.write("<body>\n")

        for j in range(len(lists)):
            line = lists[j]
            for i in range(len(line)):
                out.write(f"<img src={line[i]} alt={i}>\n")
            out.write("<hr>\n")

        out.write("</body>\n")
        out.write("</html>\n")


if __name__ == '__main__':
    MAX_CLUSTERS = 100
    paths = get_paths(sys.argv[1])
    print(len(paths))
    images = get_images(paths)


    X = normalize_images(images) # flattened vectors with pixel values

    pca = PCA(n_components=0.98)
    X = pca.fit_transform(X)
    new_n = np.cumsum(len(pca.explained_variance_ratio_))
    print("new number of components:", new_n)

    # find the best number of clusters for the K-Means algorithm
    sse = []
    list_k = list(range(2, MAX_CLUSTERS, 20))
    best_sscore = 0
    second_best_ss = 0
    best_k = 2
    second_best_k = 2
    for k in list_k:
        km = KMeans(n_clusters=k, n_init='auto')
        km.fit(X)
        ss = silhouette_score(X, km.labels_)
        if ss >= best_sscore:
            second_best_ss = best_sscore
            second_best_k = best_k
            best_sscore = ss
            best_k = k
        elif ss >= second_best_ss:
            second_best_ss = ss
            second_best_k = k       

    print("best K: ", best_k)
    print("second best K: ", second_best_k)

    # narrow down the search for the optimal number of clusters
    fromk = min(second_best_k, best_k)
    fromk = max(2, fromk -10)
    tok = max(second_best_k, best_k)
    tok = min(MAX_CLUSTERS -1, tok+10)
    list_k2 = list(range(fromk, tok))
    for k in list_k2:
        km = KMeans(n_clusters=k, n_init='auto')
        km.fit(X)
        ss = silhouette_score(X, km.labels_)
        if ss >= best_sscore:
            best_sscore = ss
            best_k = k

    print("best number of clusters for the K-Means algorithm:", best_k)

    km = KMeans(n_clusters=best_k, n_init='auto')
    km.fit(X)
    labels = km.labels_
    lists = create_output_list(labels, paths)
    create_output_files(lists)
    print(len(lists))
