## Audio/Music Classification using Unsupervised Learning 

I have attached four .ipynb notebooks each of them concentrating and explaining one topic.

#### 1. Data Visualization task: 
This notebook consist of feature extraction of song example considered 'American Idiot' by Green Days. It uses librosa library and matplotlib function to display visual representation
Link towards the jupyter notebook of Data Visualization: https://drive.google.com/file/d/1M3bTM2zDmLZKgPNo2r-GT1IayQhaNJUo/view?usp=sharing

#### 2. Clustering_Implementation: 
This notebook handles multiple songs and generates all the features from it. The 11 features generated are Zero Cross Rate, Energy, Entropy of Energy, Spectral Centroid, Spectral Spread, Spectral Entropy, Spectral Flux, Spectral Roll off, MFCC, Chroma Vector, Chroma Deviation

Using feature_extractor function we have generated that from pyAudioAnalysis library.

Further selecting the Chromogram features only which is a powerful representation of music into 12 octaves. And based on these features of each song. Using Kmeans clustering, we can group the songs into similar clusters. For finding the optimal k, I have used the elbow method using scikit-learn library.

#### 3. Random_Split: 
The random_split notebook, focuses on only one song. I was confused about whether we have to perform clustering on one song only and generate important segments from it that are kind of similar or to work on multiple songs. So in this notebook what I did was to split the song into multiple parts of 60s and applied clustering on it to group them into similar clusters.

Based on the splitted music wav. One can perform clustering on these different parts of a single song. And one can find interesting clusters which actually makes sense when you listen to the song. For example, one cluster of songs would have similar kind of tone, while other's will have signature parts of the song which is chorus. I experimented with Miley Cyrus Party in USA.

#### 4. Spleeter: 
The spleeter notebook consist of the spleeter library.

The Spleeter library (based on ML library TensorFlow) released by the music streaming app Deezer is a source separation tool which attempts to separate the different elements of a piece of audio so that you can hear those elements separate from the whole mixed track.

As mentioned one can also use spleeter to split the audio into vocals and music separately. I found out about this library and thought it was quite interesting. Hence isntead of manually going through each and every parts of songs and applying clustering on it one can use these libraries as well.
