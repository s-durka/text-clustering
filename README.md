# **Image Clustering using K-Means**  

## **Problem Description**  
The purpose of this program is to **cluster images** of various characters using machine learning techniques. The dataset contains images of different resolutions, depicting characters with varying styles, thickness, and sizes.

## 📂 **Dataset Details**  
🔗 **Dataset Link:** [Download Here](https://www.dropbox.com/scl/fo/wp4iz69odzi8ldnplwp0h/h?rlkey=udboc94aueqzs5rom9fpcjfvm&dl=0)  

### 🔍 **Challenges in the Dataset:**  
- Images include individual characters, character sequences, and fragments.  
- The number of **decision classes** is unknown due to the presence of incomplete or joined letters.  
- Some letters may be **indistinguishable**, such as:  
  - `"I"`, `"|"`, and `"1"`  
  - `"m"` and `"rn"`  

---

## **Solution Approach**  
This solution utilizes the **K-Means clustering algorithm** to group similar images.  

### **Steps Involved:**  
1️⃣ **Preprocessing:**  
   - Convert images to **grayscale** numpy arrays (values: `0-255`).  
   - Resize images to a fixed dimension.  
   - Flatten each 2D image into a **1D feature vector**.  

2️⃣ **Dimensionality Reduction:**  
   - Apply **Principal Component Analysis (PCA)** to reduce feature space.  

3️⃣ **Clustering with K-Means:**  
   - Determine the **optimal number of clusters (n)** by evaluating **silhouette scores** from multiple runs.  
   - Perform clustering using **Euclidean distance** as the metric.  

### ** Balancing Clustering Issues:**  
- If **n is too low**, the same letter may be split into different clusters.  
- If **n is too high**, different letters may be grouped incorrectly.  
- This approach aims to **find a balance** between these issues.  

---

## **Run the Program**  

```bash
# (in a Python virtual environment)
# Install dependencies
pip install -r requirements.txt

# Run the clustering script
python3 ./cluster.py <path-to-dataset>
