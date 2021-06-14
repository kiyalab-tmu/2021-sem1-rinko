import sys
import pandas as pd
import matplotlib.pyplot as plt

args = sys.argv

def main():
    df = pd.read_csv('./log/'+args[1])

    x = df['epoch'].values
    y0 = df['train_loss'].values
    y1 = df['test_loss'].values
    y2 = df['train_acc'].values
    y3 = df['test_acc'].values

    fig = plt.figure()
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(2,1,1)
    #ax1.xlabel('epoch')
    ax1.plot(x, y0, label='train_loss')
    ax1.plot(x, y1, label='test_loss')
    ax1.legend()

    ax2 = fig.add_subplot(2,1,2)
    #ax2.xlabel=('epoch')
    ax2.plot(x, y2, label='train_acc')
    ax2.plot(x, y3, label='test_acc')
    ax2.legend()

    plt.show()

    fig.savefig('./outputs/'+(args[1]).split('.')[0]+'_curve.png')

if __name__ == '__main__':
    main()

