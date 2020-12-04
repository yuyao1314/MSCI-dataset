import cv2
import numpy as np
import os


image_paths = []
path = ".../dataset"


#list of our class names
training_names = os.listdir(path)


training_paths = []
names_path = []
#get full list of all training images
for p in training_names:
    training_paths1 = os.listdir("..."+p)
    for j in training_paths1:
        training_paths.append("..."+j)
        names_path.append(p)


sift = cv2.SIFT()
#surf = cv2.SURF()
print names_path


descriptors_unclustered = []


dictionarySize = 60


BOW = cv2.BOWKMeansTrainer(dictionarySize)


for p in training_paths:
    image = cv2.imread(p)
    gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp, dsc= sift.detectAndCompute(gray, None)
    BOW.add(dsc)


#dictionary created
dictionary = BOW.cluster()




FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
sift2 = cv2.DescriptorExtractor_create("SIFT")
#surf2 = cv2.DescriptorExtractor_create("SURF")
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction = cv2.BOWImgDescriptorExtractor(surf2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)
print "bow dictionary", np.shape(dictionary)




#returns descriptor of image at pth
def feature_extract(pth):
    im = cv2.imread(pth, 1)
    gray = cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return bowDiction.compute(gray, sift.detect(gray))


train_desc = []
train_labels = []
i = 0
for p in training_paths:
    train_desc.extend(feature_extract(p))
    if names_path[i]=='normal':
        train_labels.append(0)
    if names_path[i]=='CIN1':
        train_labels.append(1)
    if names_path[i]=='CIN2/3':
        train_labels.append(2)
    if names_path[i]=='Cancer':
        train_labels.append(3)


    i = i+1
print(train_desc)
print "svm items", len(train_desc), len(train_desc[0])
count=0
svm = cv2.SVM()
svm.train(np.array(train_desc), np.array(train_labels))
i=0
j=0


confusion = np.zeros((2,2))
def classify(pth):
    feature = feature_extract(pth)
    p = svm.predict(feature)
    w = int(p)
    confusion[train_labels[count]-1, w-1] = confusion[train_labels[count]-1, w-1] + 1
    
    


for p in training_paths:
    #print(p)
    classify(p)
    count += 1
print(confusion)
def normalizeRows(M):
    row_sums = M.sum(axis=1)
    return M / row_sums
    
confusion = normalizeRows(confusion)


confusion = confusion.transpose()
    
print confusion