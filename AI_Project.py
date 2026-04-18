import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


###########################image loading function################################
def load_images(data_dir, size=(32, 32)):
    X, y = [], []
    for label in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, label)
        if not os.path.isdir(class_folder): continue
        for file in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file)
            try:
                img = Image.open(file_path).convert('L').resize(size)
                X.append(np.array(img).flatten())
                y.append(label)
            except:
                pass
    return np.array(X), np.array(y)


#path to the dataset
data_path = "C:/Users/HP/Downloads/archive/seg_train/seg_train1"
X, y = load_images(data_path)
X = X / 255.0  #normalize pixel values to [0, 1]


#encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


#split the data into: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


#train naive bayes model
nb_model = GaussianNB().fit(X_train, y_train)


#train decision tree model
dt_model = DecisionTreeClassifier().fit(X_train, y_train)


#train MLP neural net model with two hidden layers: 128 and 64 neurons
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42).fit(X_train, y_train)


##############################print format function##################################
def print_header(title):
    print(f"\n{title}\n{'=' * 40}")


##########################printing precision, recall, and F1-score for each class############################3
def print_metrics(model_name, y_true, y_pred, encoder, target_classes):
    print_header(model_name)
    for cls in target_classes:
        idx = list(encoder.classes_).index(cls)
        precision = precision_score(y_true, y_pred, labels=[idx], average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=[idx], average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[idx], average='macro', zero_division=0)
        print(f"Class: {cls}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall   : {recall:.2f}")
        print(f"F1-score : {f1:.2f}\n")


############################printing confusion matrix function##################################3
def print_confusion_matrix(model_name, y_true, y_pred, encoder):
    print_header(f"{model_name} Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    labels = list(encoder.classes_)

    #print header row
    header = " " * 10 + "".join(f"{label:>10}" for label in labels)
    print(header)
    print("-" * len(header))

    #print each row
    for i, row in enumerate(cm):
        row_label = f"{labels[i]:<10}"
        values = "".join(f"{val:>10}" for val in row)
        print(row_label + values)
    print()


#accuracy comparison
print_header("Accuracy Comparison")
print(f"Naive Bayes     : {accuracy_score(y_test, nb_model.predict(X_test)):.4f}")
print(f"Decision Tree   : {accuracy_score(y_test, dt_model.predict(X_test)):.4f}")
print(f"MLP Neural Net  : {accuracy_score(y_test, mlp_model.predict(X_test)):.4f}")


#per-class metrics
target_classes = label_encoder.classes_
print_metrics("Naive Bayes", y_test, nb_model.predict(X_test), label_encoder, target_classes)
print_metrics("Decision Tree", y_test, dt_model.predict(X_test), label_encoder, target_classes)
print_metrics("MLP Neural Net", y_test, mlp_model.predict(X_test), label_encoder, target_classes)


#print confusion matrices
print_confusion_matrix("Naive Bayes", y_test, nb_model.predict(X_test), label_encoder)
print_confusion_matrix("Decision Tree", y_test, dt_model.predict(X_test), label_encoder)
print_confusion_matrix("MLP Neural Net", y_test, mlp_model.predict(X_test), label_encoder)