import matplotlib.pyplot as plt


def plot_history(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    train_acc = history_dict['acc']
    test_acc = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(8, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, '-', label='Training Acc')
    plt.plot(epochs, test_acc, '-', label='Validation Acc')
    plt.title("Training And Validation Acc")
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()


def plot_regression_his(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    train_mae = history_dict['mean_absolute_error']
    test_mae = history_dict['val_mean_absolute_error']
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(8, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mae, '-', label='Training Mae')
    plt.plot(epochs, test_mae, '-', label='Validation Mae')
    plt.title("Training And Validation Acc")
    plt.xlabel('Epochs')
    plt.ylabel('Mae')
    plt.legend()
    plt.show()