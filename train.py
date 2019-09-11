from data_loader.data_gen import *
from models.unet import *
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def plot_history(history,model):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('data/results/loss_history_'+model)
def train(args):
    data_gen_args = dict(rotation_range=0.3,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')
    data_gen = Data_generator()
    size = int(args['size'])
    target_size = (size,size)
    train_data = data_gen.train_gen(data_gen_args,batch_size=2,target_size=target_size)
    val_data = data_gen.val_gen(target_size=target_size)
    if args['model']=='unet':
        unet = Unet(input_size=target_size)
    elif args['model']=='attention':
        unet = Attention_unet(input_size=target_size)
    elif args['model']=='dense':
        unet = Dense_unet(input_size=target_size)
        unet.save=False
    history = unet.train(train_data,val_gen=val_data,steps_per_epoch=int(args['step']),epochs=int(args['epoch']))
    plot_history(history,args['model'])



if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',choices=['unet','attention','dense'],default='attention')
    parser.add_argument('--size',default=256)
    parser.add_argument('--epoch',default=50)
    parser.add_argument('--step',default=500)
    args=vars(parser.parse_args())
    train(args)
