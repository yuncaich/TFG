from prettytable import PrettyTable
import matplotlib.pyplot as plt

class ConfusionMatrix(object):

  def __init__(self, num_classes: int, labels: list):
    self.matrix = np.zeros((num_classes,num_classes))
    self.num_classes = num_classes
    self.labels = labels

  def update(self, preds, labels):
    for p, t in zip(preds, labels):
      p = p.astype(int)
      t = t.astype(int)
      self.matrix[t,p] += 1

  def summary(self):
    #calcular accuracy
    sum_TP = 0
    for i in range(self.num_classes):
      sum_TP += self.matrix[i,i]

    acc = sum_TP/np.sum(self.matrix)
    print("The model accuracy is" , acc)

     #Presicion, recal, specificity
    table = PrettyTable()
    table.field_names = ["", "Presicion", "Recall", "Specificity"]

    for i in range (self.num_classes):
      TP = self.matrix[i,i]
      FP = np.sum(self.matrix[i, :]) - TP
      FN = np.sum(self.matrix[:, i]) - TP
      TN = np.sum(self.matrix) - TP - FP - FN

      Presicion = round(TP / (TP + FP), 3)
      Recall = round(TP / (TP + FN), 3)
      Specificity = round(TN / (TN + FP), 3)

      table.add_row([self.labels[i],Presicion,Recall,Specificity])

    print(table)

  def plot(self):
    matrix = self.matrix

    print(matrix)

    plt.imshow(matrix , cmap=plt.cm.Blues)

    #X label
    plt.xticks(range(self.num_classes) , self.labels, rotation=90 , size = "small")

    #y label

    plt.yticks(range(self.num_classes) , self.labels)

    #显示 colorbar
    plt.colorbar()
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Confusion matrix")

      #标注数量 & 概率信息

    thresh = matrix.max() / 2
    for x in range (self.num_classes):  #columnas
      for y in range(self.num_classes):  #fila
        info = int(matrix[y,x])
        plt.text(x, y, info,                    #escribir valor
                    verticalalignment ='center',
                    horizontalalignment ='center',
                    color = "white" if info > thresh else "black")

    ##plt.tight_layout()
    plt.show()

