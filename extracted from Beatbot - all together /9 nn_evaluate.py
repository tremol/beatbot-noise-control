## Evaluate model quality
# Check the accuracy of the model against the testing and training sets, and compute the confusion matrix.
import itertools
from sklearn.metrics import confusion_matrix
def accuracy_rating(net, dataloader, dataset_label):
    """ Print the fraction of correct predictions on a data loader. """
    correct = 0
    total = 0
    all_targets = torch.tensor([], dtype=torch.long)
    all_predictions = torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        for data in dataloader:
            spectrograms, labels = data
            outputs = net(spectrograms)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions = torch.cat( 
                (all_predictions, predicted), dim=0 )
            all_targets = torch.cat( 
                (all_targets, labels), dim=0 )
            
    print('Accuracy of the network on the {} {} spectrograms: {:.0f} %'.format(
        total,
        dataset_label,
        100 * correct / total))
    
    return all_predictions, all_targets
# adapted from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(predictions, targets, labels_int_to_str, 
                          normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """ Compute and display the confusion matrix. """
    
    stacked = torch.stack( [targets, predictions], dim=1 )
    all_int_labels = sorted(list(labels_int_to_str.keys()))
    num_labels = len(all_int_labels)
    confusion_matrix = torch.zeros(num_labels, num_labels, dtype=torch.int64)

    for pair in stacked:
        target_label, prediction_label = pair.tolist()
        confusion_matrix[target_label, prediction_label] += 1

    classes = [ labels_int_to_str[i] for i in all_int_labels ]    
    cm = confusion_matrix # rename for compactness
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization:')
        pass
        
    size = min(0.7 * (num_labels + 1), 8)
    plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return confusion_matrix
# # TESTING
# accuracy_rating(my_net, my_train_loader, 'training')
# preds, targets = accuracy_rating(my_net, my_test_loader, 'test')

# plot_confusion_matrix(preds, targets, my_dataset.noise_int_to_str);